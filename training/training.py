import json
import torch
import torch.nn as nn

from pytorch_msssim import SSIM
from torchvision.utils import save_image


class Trainer:
    """Class used to train neural renderers.

    Args:
        device (torch.device): Device to train model on.
        model (models.neural_renderer.NeuralRenderer): Model to train.
        lr (float): Learning rate.
        rendering_loss_type (string): One of 'l1', 'l2'.
        ssim_loss_weight (float): Weight assigned to SSIM loss.
    """
    def __init__(self, device, model, lr=2e-4, rendering_loss_type='l1',
                 ssim_loss_weight=0.05, feature_loss=False, iteration_verbose=1):
        self.device = device
        self.model = model
        self.lr = lr
        self.rendering_loss_type = rendering_loss_type
        self.ssim_loss_weight = ssim_loss_weight
        self.use_ssim = self.ssim_loss_weight != 0
        # If False doesn't save losses in loss history
        self.register_losses = True
        # Check if model is multi-gpu
        self.multi_gpu = isinstance(self.model, nn.DataParallel)
        self.iteration_verbose = iteration_verbose
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Initialize loss functions
        # For rendered images
        if self.rendering_loss_type == 'l1':
            self.loss_func = nn.L1Loss()
        elif self.rendering_loss_type == 'l2':
            self.loss_func = nn.MSELoss()
        self.feature_loss = feature_loss
        self.scene_feature_loss = nn.L1Loss()

        # For SSIM
        if self.use_ssim:
            self.ssim_loss_func = SSIM(data_range=1.0, size_average=True,
                                       channel=3, nonnegative_ssim=False)

        # Loss histories
        self.recorded_losses = ["total", "regression", "ssim"]
        self.loss_history = {loss_type: [] for loss_type in self.recorded_losses}
        self.epoch_loss_history = {loss_type: [] for loss_type in self.recorded_losses}
        self.val_loss_history = {loss_type: [] for loss_type in self.recorded_losses}

        self.image_weight = 0.9
        self.scene_weight = 0.1
        self.after_scene_epoch = 10

    def train(self, dataloader, epochs, save_dir=None, save_freq=1,
              test_dataloader=None, load_path=None, resume_epoch=None):
        """Trains a neural renderer model on the given dataloader.

        Args:
            dataloader (torch.utils.DataLoader): Dataloader for a
                misc.dataloaders.SceneRenderDataset instance.
            epochs (int): Number of epochs to train for.
            save_dir (string or None): If not None, saves model and generated
                images to directory described by save_dir. Note that this
                directory should already exist.
            save_freq (int): Frequency with which to save model.
            test_dataloader (torch.utils.DataLoader or None): If not None, will
                test model on this dataset after every epoch.
            base_model (models.neural_renderer.NeuralRenderer): resume to train.

        """
        if save_dir is not None:
            # Save model after training

            if load_path is not None:
                base_experiment_name = load_path.split("/")[1]
                if self.multi_gpu:
                    self.model.module.save(save_dir + "/"+base_experiment_name+"_base_model.pt")
                else:
                    self.model.save(save_dir + "/"+base_experiment_name+"_base_model.pt")
            if resume_epoch:
                print("Resuming epoch:", resume_epoch+1)
            # Extract one batch of data
            for batch in dataloader:
                break
            # Save original images
            save_image(batch["img"], save_dir + "/imgs_ground_truth.png", nrow=4)
            # Store batch to check how rendered images improve during training
            self.fixed_batch = batch
            # Render images before any training
            rendered = self._render_fixed_img()
            save_image(rendered.detach(),
                       save_dir + "/imgs_gen_{}.png".format(str(0).zfill(3)), nrow=4)
            for batch in test_dataloader:
                break
            save_image(batch["img"], save_dir + "/imgs_test_ground_truth.png", nrow=4)
            self.fixed_test_batch = batch
            rendered = self._render_fixed_test_img()
            save_image(rendered.detach(),
                       save_dir + "/imgs_test_gen_{}.png".format(str(0).zfill(3)), nrow=4)

        for epoch in range(epochs):
            self.epoch = epoch
            epoch = epoch + resume_epoch if resume_epoch is not None else epoch
            print("\nEpoch {}".format(epoch + 1)) if (epoch + 1) % self.iteration_verbose == 0 else None
            self._train_epoch(dataloader)
            # Update epoch loss history with mean loss over epoch
            for loss_type in self.recorded_losses:
                self.epoch_loss_history[loss_type].append(
                    sum(self.loss_history[loss_type][-len(dataloader):]) / len(dataloader)
                )
            # Print epoch losses
            print("Mean epoch loss:")
            self._print_losses(epoch_loss=True)

            # Optionally save generated images, losses and model
            if save_dir is not None:
                # Save generated images
                with torch.no_grad():
                    rendered = self._render_fixed_img()
                    save_image(rendered.detach(),
                               save_dir + "/imgs_gen_{}.png".format(str(epoch + 1).zfill(3)), nrow=4)
                    rendered = self._render_fixed_test_img()
                    save_image(rendered.detach(),
                               save_dir + "/imgs_test_gen_{}.png".format(str(epoch + 1).zfill(3)), nrow=4)
                    # Save losses
                    with open(save_dir + '/loss_history.json', 'w') as loss_file:
                        json.dump(self.loss_history, loss_file)
                    # Save epoch losses
                    with open(save_dir + '/epoch_loss_history.json', 'w') as loss_file:
                        json.dump(self.epoch_loss_history, loss_file)
                    # Save model
                    if (epoch + 1) % save_freq == 0:
                        if self.multi_gpu:
                            self.model.module.save(save_dir + "/model.pt")
                        else:
                            self.model.save(save_dir + "/model.pt")

                if test_dataloader is not None:
                    regression_loss, ssim_loss, total_loss = mean_dataset_loss(self, test_dataloader)
                    print("Validation:\nRegression: {:.4f}, SSIM: {:.4f}, Total: {:.4f}".format(regression_loss, ssim_loss, total_loss))
                    self.val_loss_history["regression"].append(regression_loss)
                    self.val_loss_history["ssim"].append(ssim_loss)
                    self.val_loss_history["total"].append(total_loss)
                    if save_dir is not None:
                        # Save validation losses
                        with open(save_dir + '/val_loss_history.json', 'w') as loss_file:
                            json.dump(self.val_loss_history, loss_file)
                        # If current validation loss is the lowest, save model as best
                        # model
                        if min(self.val_loss_history["total"]) == total_loss:
                            print("New best model!")
                            if self.multi_gpu:
                                self.model.module.save(save_dir + "/best_model.pt")
                            else:
                                self.model.save(save_dir + "/best_model.pt")

        # Save model after training
        if save_dir is not None:
            if self.multi_gpu:
                self.model.module.save(save_dir + "/model.pt")
            else:
                self.model.save(save_dir + "/model.pt")

    def _train_epoch(self, dataloader):
        """Trains model for a single epoch.

        Args:
            dataloader (torch.utils.DataLoader): Dataloader for a
                misc.dataloaders.SceneRenderDataset instance.
        """
        num_iterations = len(dataloader)
        for i, batch in enumerate(dataloader):
            # Train inverse and forward renderer on batch
            # torch.cuda.empy_cache()
            self._train_iteration(batch)

            # Print iteration losses
            print("{}/{}".format(i + 1, num_iterations))
            self._print_losses()

    def _train_iteration(self, batch):
        """Trains model for a single iteration.

        Args:
            batch (dict): Batch of data as returned by a Dataloader for a
                misc.dataloaders.SceneRenderDataset instance.
        """
        imgs, rendered, scenes, scenes_rotated = self.model(batch)
        self._optimizer_step(imgs, rendered, scenes, scenes_rotated)

    def _optimizer_step(self, imgs, rendered, scenes, scenes_rotated):
        """Updates weights of neural renderer.

        Args:
            imgs (torch.Tensor): Ground truth images. Shape
                (batch_size, channels, height, width).
            rendered (torch.Tensor): Rendered images. Shape
                (batch_size, channels, height, width).
        """
        self.optimizer.zero_grad()

        loss_regression = self.loss_func(rendered, imgs)

        if self.feature_loss and self.epoch > self.after_scene_epoch:
            # swapped_idx = get_swapped_indices(scenes_rotated.shape[0])
            scene_loss = self.scene_feature_loss(scenes, scenes_rotated) * self.scene_weight
            print("Scene Loss: " + str(scene_loss.detach().item()))
            loss_regression = loss_regression * self.image_weight + scene_loss
            # loss_regression = loss_regression+ scene_loss

        if self.use_ssim:
            # We want to maximize SSIM, i.e. minimize -SSIM
            loss_ssim = 1. - self.ssim_loss_func(rendered, imgs)
            loss_total = loss_regression + self.ssim_loss_weight * loss_ssim
        else:
            loss_total = loss_regression

        loss_total.backward()
        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
        self.optimizer.step()

        # Record total loss
        if self.register_losses:
            self.loss_history["total"].append(loss_total.detach().item())
            self.loss_history["regression"].append(loss_regression.detach().item())
            # If SSIM is not used, register 0 in logs
            if not self.use_ssim:
                self.loss_history["ssim"].append(0.)
            else:
                self.loss_history["ssim"].append(loss_ssim.detach().item())

    def _render_fixed_img(self):
        """Reconstructs fixed batch through neural renderer (by inferring
        scenes, rotating them and rerendering).
        """
        _, rendered, _, _ = self.model(self.fixed_batch)
        return rendered

    def _render_fixed_test_img(self):
        """Reconstructs fixed batch through neural renderer (by inferring
        scenes, rotating them and rerendering).
        """
        _, rendered, _, _ = self.model(self.fixed_test_batch)
        return rendered

    def _print_losses(self, epoch_loss=False):
        """Prints most recent losses."""
        loss_info = []
        for loss_type in self.recorded_losses:
            if epoch_loss:
                loss = self.epoch_loss_history[loss_type][-1]
            else:
                loss = self.loss_history[loss_type][-1]
            loss_info += [loss_type, loss]
        print("{}: {:.3f}, {}: {:.3f}, {}: {:.3f}".format(*loss_info))


def mean_dataset_loss(trainer, dataloader):
    """Returns the mean loss of a model across a dataloader.

    Args:
        trainer (training.Trainer): Trainer instance containing model to
            evaluate.
        dataloader (torch.utils.DataLoader): Dataloader for a
            misc.dataloaders.SceneRenderDataset instance.
    """
    # No need to calculate gradients during evaluation, so disable gradients to
    # increase performance and reduce memory footprint
    with torch.no_grad():
        # Ensure calculated losses aren't registered as training losses
        trainer.register_losses = False

        regression_loss = 0.
        ssim_loss = 0.
        total_loss = 0.
        for i, batch in enumerate(dataloader):
            imgs, rendered, scenes, scenes_rotated = trainer.model(batch)

            # Update losses
            # Use _loss_func here and not _loss_renderer since we only want regression term
            current_regression_loss = trainer.loss_func(rendered, imgs).item()
            if trainer.feature_loss and trainer.epoch > trainer.after_scene_epoch:
                current_regression_loss = current_regression_loss * trainer.image_weight + \
                                          trainer.scene_feature_loss(scenes, scenes_rotated).item() * trainer.scene_weight
            if trainer.use_ssim:
                current_ssim_loss = 1. - trainer.ssim_loss_func(rendered, imgs).item()
            else:
                current_ssim_loss = 0.
            regression_loss += current_regression_loss
            ssim_loss += current_ssim_loss
            total_loss += current_regression_loss + trainer.ssim_loss_weight * current_ssim_loss

        # Average losses over dataset
        regression_loss /= len(dataloader)
        ssim_loss /= len(dataloader)
        total_loss /= len(dataloader)

        # Reset boolean so we register losses if we continue training
        trainer.register_losses = True

    return regression_loss, ssim_loss, total_loss
