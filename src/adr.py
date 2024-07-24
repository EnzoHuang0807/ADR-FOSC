import torch
import torch.nn.functional as F

from util.utils import cosine_scheduler


class ADR:
    def __init__(
        self,
        num_classes,
        total_epoch,
        train_loader_length,
        temperature_low,
        temperature_high,
        interpolation_low,
        interpolation_high,
        fosc_threshold
    ):
        self.num_classes = num_classes
        self.temperature_scheduler = cosine_scheduler(
            temperature_high,
            temperature_low,
            total_epoch,
            train_loader_length,
        )
        self.interpolation_scheduler = cosine_scheduler(
            interpolation_low,
            interpolation_high,
            total_epoch,
            train_loader_length,
        )

        self.fosc_threshold = fosc_threshold
        self.interpolation = 0


    def __call__(self, img, label, teacher_model, step, model, train_mode, attacker):
        with torch.no_grad():
            one_hot_label = torch.nn.functional.one_hot(
                label, num_classes=self.num_classes
            )
            logits_teacher = teacher_model(img)
            logits_teacher /= self.temperature_scheduler[step]
            # logits_teacher /= 3
            prob_teacher = F.softmax(logits_teacher, dim=1)

            target_prob = torch.gather(prob_teacher, 1, label.unsqueeze(1)).squeeze()
            max_prob, _ = torch.max(prob_teacher, 1)


        cls_loss_fn = torch.nn.CrossEntropyLoss()
        left_bound = 0
        right_bound = 20

        if self.fosc_threshold is None:
            
            prob_per_img = self.interpolation_scheduler[step] - (max_prob - target_prob)
            # prob_per_img = 0.7 - (max_prob - target_prob)
            prob_per_img = torch.clamp(prob_per_img, min=0.0, max=1.0)
            prob_per_img = prob_per_img.unsqueeze(1)
            rectified_label = (
                prob_per_img * prob_teacher + (1 - prob_per_img) * one_hot_label
            )

        else:

            while left_bound < right_bound:

                middle = (left_bound + right_bound) // 2
                self.interpolation = middle * 0.05

                prob_per_img = self.interpolation - (max_prob - target_prob)
                prob_per_img = torch.clamp(prob_per_img, min=0.0, max=1.0)
                prob_per_img = prob_per_img.unsqueeze(1)
                rectified_label = (
                    prob_per_img * prob_teacher + (1 - prob_per_img) * one_hot_label
                )
            
                img_adv = attacker.attack(model, img, rectified_label)
                img_adv.requires_grad_(True)
                grad = torch.autograd.grad(
                        cls_loss_fn(model(img_adv), rectified_label), img_adv, 
                        retain_graph=False, create_graph=False
                )[0]

                fosc = torch.sum((-1 * torch.einsum("ijkl, ijkl -> i", (img_adv - img), grad) 
                        + 8.0 / 255 *  torch.norm(grad, dim = (1,2,3), p = 1))).item()

                if fosc <= self.fosc_threshold:
                    right_bound = middle
                else:
                    left_bound = middle + 1   


        return rectified_label
