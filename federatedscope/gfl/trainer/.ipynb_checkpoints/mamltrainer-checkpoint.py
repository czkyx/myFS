from federatedscope.register import register_trainer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
import os
import numpy as np
import learn2learn as l2l

class GraphMAMLTrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
       # maml = l2l.algorithms.MAML(self.ctx.model, lr=self.cfg.maml.inner_lr)
        maml = l2l.algorithms.MAML(self.ctx.model, lr= 0.01)
        ctx.maml = maml.clone()

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        if self.cfg.model.task.endswith('Regression'):
            label = batch.y.float()
        else:
            label = batch.y.squeeze(-1).long()

        if ctx.get("finetune", False):
            # update on the model
            pred = ctx.model(batch)
        else:
            # update on the clone
            pred = ctx.maml(batch)
        ctx.loss_batch = ctx.criterion(pred, label)

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred

    def _hook_on_batch_backward(self, ctx):
        if ctx.get("finetune", False):
            # normal update when finetune
            ctx.optimizer.zero_grad()
            ctx.loss_batch.backward()
            ctx.optimizer.step()
        else:
            # during train, fake forward
            ctx.maml.adapt(ctx.loss_batch, allow_unused=True, allow_nograd=True)  ## add allow_nograd=True !!!!!!!!!!!!!!!

    def _hook_on_batch_end(self, ctx):
        # keep the last batch here
        data_batch = ctx.data_batch
        super()._hook_on_batch_end(ctx)
        ctx.data_batch = data_batch

    def _hook_on_fit_end(self, ctx):
        if ctx.cur_mode == "train" and not ctx.get("finetune", False):
            # outer loop, reuse the last batch
            batch = ctx.data_batch.to(ctx.device)
            if self.cfg.model.task.endswith('Regression'):
                label = batch.y.float()
            else:
                label = batch.y.squeeze(-1).long()
            # forward
            pred_outer = ctx.maml(batch)
            ctx.loss_batch = ctx.criterion(pred_outer, label)

            ctx.optimizer.zero_grad()
            ctx.loss_batch.backward()
            ctx.optimizer.step()

        ctx.data_batch = None
        ctx.maml = None
        super()._hook_on_fit_end(ctx)
    
    ## change   ####################
    def save_prediction(self, path, client_id, task_type):
        y_inds, y_probs = self.ctx.test_y_inds, self.ctx.test_y_prob
        os.makedirs(path, exist_ok=True)

        # TODO: more feasible, for now we hard code it for cikmcup
        y_preds = np.argmax(y_probs, axis=-1) if 'classification' in task_type.lower() else y_probs

        if len(y_inds) != len(y_preds):
            raise ValueError(f'The length of the predictions {len(y_preds)} not equal to the samples {len(y_inds)}.')

        with open(os.path.join(path, 'prediction.csv'), 'a') as file:
            for y_ind, y_pred in zip(y_inds,  y_preds):
                if 'classification' in task_type.lower():
                    line = [client_id, y_ind] + [y_pred]
                else:
                    line = [client_id, y_ind] + list(y_pred)
                file.write(','.join([str(_) for _ in line]) + '\n')

def call_graph_level_trainer(trainer_type):
    if trainer_type == 'graphmaml_trainer':
        trainer_builder = GraphMAMLTrainer
        return trainer_builder


register_trainer('graphmaml_trainer', call_graph_level_trainer)