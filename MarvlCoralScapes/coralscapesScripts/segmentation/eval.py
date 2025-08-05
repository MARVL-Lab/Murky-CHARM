import numpy as np 

import torch 

from transformers.image_processing_base import BatchFeature

from torchmetrics.classification import Accuracy, JaccardIndex
from torchmetrics.segmentation import MeanIoU

from coralscapesScripts.visualization import denormalize_image, color_label
from coralscapesScripts.datasets.preprocess import preprocess_batch, get_windows

def get_batch_predictions_eval(data, model, device, preprocessor = None):
    """
    Generate batch predictions for evaluation using different types of models.
    Args:
        data (tuple, list, or dict): Input data for the model. It can be:
            - A tuple or list containing inputs and labels for CNN & DINO models.
            - A dictionary containing "pixel_values" and "labels" for Segformer models.
        model (torch.nn.Module): The model used for generating predictions.
        device (torch.device): The device (CPU or GPU) on which the computation will be performed.
        preprocessor (optional): A preprocessor object for post-processing the outputs of Segformer models.
    Returns:
        torch.Tensor: The predicted outputs from the model.
    """

    if(isinstance(data,tuple) or isinstance(data,list)): #For CNN & DINO models
        inputs, labels = data 
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()

        outputs = model(inputs)
        if(hasattr(outputs, "logits")): #For DINO models
            outputs = outputs.logits
            outputs = torch.nn.functional.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        outputs = outputs.argmax(dim=1)

    if("pixel_values" and "labels" in data):  #For Segformer models
        # print("data:----------------", data["labels"])
        # target_sizes = [
        #     (image.shape[1], image.shape[2]) for image in data["labels"]
        # ]
        target_sizes = [
            (image.shape[0], image.shape[1]) for image in data["labels"]  # added by shruti
        ]
        outputs = model(
            pixel_values=data["pixel_values"].to(device),
        )
        outputs = preprocessor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
        outputs = torch.stack(outputs)

    if(isinstance(outputs, dict) and "out" in outputs):
        outputs = outputs["out"]

    return outputs

# below function added by shruti 


# import torch.nn.functional as F

# def get_batch_predictions_eval(data, model, device, preprocessor=None):
#     print("data:--------------", data)
#     outputs = None
#     loss = None

#     if isinstance(data, (tuple, list)):  # For CNN & DINO models
#         inputs, labels = data 
#         print("inputs:----------", inputs)
#         print("labels:----------", labels)
#         inputs = inputs.to(device).float()
#         labels = labels.to(device).long()

#         model_out = model(inputs)
#         if hasattr(model_out, "logits"):  # DINO case
#             logits = model_out.logits
#             logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
#         else:
#             logits = model_out

#         outputs = logits.argmax(dim=1)
#         loss = F.cross_entropy(logits, labels)

#     elif isinstance(data, dict) and "pixel_values" in data:  # For Segformer
#         pixel_values = data["pixel_values"].to(device)
#         print("pixel_values:---------------", pixel_values)
#         labels = data.get("labels", None)

#         model_out = model(pixel_values=pixel_values)

#         if preprocessor is not None and labels is not None:
#             try:
#                 target_sizes = [(label.shape[-2], label.shape[-1]) for label in labels]
#             except Exception as e:
#                 print("[ERROR] Invalid label shape in target_sizes:", [label.shape for label in labels])
#                 raise e
#             outputs = preprocessor.post_process_semantic_segmentation(model_out, target_sizes=target_sizes)
#             outputs = torch.stack(outputs)
#         elif hasattr(model_out, "logits"):
#             logits = model_out.logits
#             outputs = logits.argmax(dim=1)

#         if labels is not None and hasattr(model_out, "logits"):
#             labels = torch.stack(labels).to(device)
#             logits = model_out.logits
#             logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
#             loss = F.cross_entropy(logits, labels)

#     else:
#         raise ValueError("Unsupported data format passed to get_batch_predictions_eval")

#     return outputs




class Evaluator:
    def __init__(self, N_classes: int, device:torch.device = "cuda", metric_dict: dict = None, preprocessor = None, eval_params: dict = None):
        """
        Initializes the evaluation class with specified metrics. Currently usable for classification and semantic segmentation tasks.
        Args:
            N_classes (int): Number of classes for the classification task.
            device (torch.device, optional): Device to run the evaluation on. Defaults to "cuda".
            metric_dict (dict, optional): Dictionary containing the metrics to be computed. If None, default metrics are used: accuracy and mean_iou.
            preprocessor (object, optional): Preprocessor object for post-processing the segmentation outputs. Defaults to None.
            eval_params (dict, optional): Evaluation parameters for sliding window inference. Defaults to None.
        """
        self.device = device
        self.N_classes = N_classes
        if(metric_dict):
            self.metric_dict = metric_dict
        else:
            self.metric_dict = {
                                "accuracy": Accuracy(task="multiclass" if N_classes > 2 else "binary", num_classes=int(N_classes), ignore_index = 0).to(device),
                                "mean_iou": JaccardIndex(task="multiclass" if N_classes > 2 else "binary", num_classes=int(N_classes), ignore_index = 0).to(device)
                                }
        self.preprocessor = preprocessor
        self.eval = eval_params

    # Added by yx
    def convert_labels(self, inference_prediction):
        # Flip seagrass and bleached coral
        inference_prediction[inference_prediction == 4] = 100
        inference_prediction[inference_prediction == 1] = 5
        inference_prediction[inference_prediction == 100] = 1
        
        # Remove labels from 1 to 5
        inference_prediction[inference_prediction == 2] = 0
        inference_prediction[inference_prediction == 3] = 1
        inference_prediction[inference_prediction == 5] = 4
        
        # Convert labels to our dataset labels
        # Bleached:
        bleached_labels = [33,16,19,3,20,32,37] # did not include 4 as it is already converted above
        for l in bleached_labels:
            inference_prediction[inference_prediction == l] = 1
        
        # Coral:
        coral_labels = [17,27,22,34,31,25,28,6,36,21]
        for l in coral_labels:
            inference_prediction[inference_prediction == l] = 2
        
        # Coral nursery:
        nursery_labels = [12,10]
        for l in nursery_labels:
            inference_prediction[inference_prediction == l] = 3
        
        # Seafloor:
        seafloor_labels = [5,13]
        for l in seafloor_labels:
            inference_prediction[inference_prediction == l] = 4
        
        # seagrass skipped as it is already converted above

        # Convert all other labels to 0
        not_all_other_labels = [1,2,3,4,5]
        for l in range(41):
            if l not in not_all_other_labels:
                inference_prediction[inference_prediction == l] = 0
        return inference_prediction

    def evaluate_model(self, dataloader: torch.utils.data.dataloader.DataLoader, model: torch.nn.Module, split = "val", is_coralscape = False):
        """
        Evaluates the given model using the provided dataloader and computes the metrics.
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing the validation data.
            model (torch.nn.Module): The model to be evaluated.
            split (str, optional): The split of the data to be evaluated. Defaults to "val".
        Returns:
            dict: A dictionary containing the computed metric results.
        Notes:
            - The model is set to evaluation mode during the evaluation process.
            - The data is transferred to the appropriate device before making predictions.
            - The predictions are obtained by applying argmax on the model outputs.
            - The metrics are updated and computed based on the predictions and true labels.
        """

        model.eval()
        metric_results = {}
        with torch.no_grad():
            for i, vdata in enumerate(dataloader):
                # print("vdata:---------------------", vdata)
                __, vlabels = vdata
                # print("vlabels:-----------------", vlabels)
                vlabels = vlabels.to(self.device).long()
                
                if(split!="train" and self.eval and self.eval.sliding_window):
                    input_windows, label_windows = get_windows(vdata, self.eval.window, self.eval.stride, self.eval.window_target, self.eval.stride_target)
                    n_vertical, n_horizontal, batch_size = input_windows.shape[:3]
                    input_windows = input_windows.view(-1, *input_windows.shape[-3:])
                    label_windows = label_windows.view(-1, *label_windows.shape[-2:])                    
                    vdata = (input_windows, label_windows)

                preprocessed_batch = preprocess_batch(vdata, self.preprocessor)
                voutputs = get_batch_predictions_eval(preprocessed_batch, model, self.device, preprocessor=self.preprocessor)
                # print("voutputs:----", voutputs.shape)
                if is_coralscape:
                    voutputs = self.convert_labels(voutputs) # added by yx
                # print("voutputs:----", voutputs.shape)
                # voutputs, vloss = get_batch_predictions_eval(preprocessed_batch, model, self.device, preprocessor=self.preprocessor) # added by shruti


                if(split!="train" and self.eval and self.eval.sliding_window):
                    if(self.eval.window_target):
                      voutputs = voutputs.view(n_vertical, n_horizontal, batch_size, self.eval.window_target, self.eval.stride_target)     
                    else:
                      voutputs = voutputs.view(n_vertical, n_horizontal, batch_size, self.eval.window, self.eval.stride)                  
             
                    voutputs = torch.cat([torch.cat(list(row), dim=-1) for row in list(voutputs)], dim=-2)

                ## Update metrics
                for metric in self.metric_dict.values():
                    metric.update(voutputs, vlabels)
        
        ## Compute metrics
        for metric_name in self.metric_dict:
            metric_results[metric_name] = self.metric_dict[metric_name].compute().cpu().numpy()
            if(metric_results[metric_name].ndim==0):
                metric_results[metric_name] = metric_results[metric_name].item()
            self.metric_dict[metric_name].reset()
        print("metric_results:----", metric_results)
        return metric_results


    def evaluate_image(self, dataloader: torch.utils.data.dataloader.DataLoader, model: torch.nn.Module, split = "val", epoch = 0):
        """
        Evaluates the given model on one image of the dataloader.
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing the validation data.
            model (torch.nn.Module): The model to be evaluated.
            split (str, optional): The split of the data to be evaluated. Defaults to "val".
            epoch (int, optional): The epoch number for logging purposes. Defaults to 0.
        Returns:
            dict: A dictionary containing the computed metric results.
        Notes:
            - The model is set to evaluation mode during the evaluation process.
            - The data is transferred to the appropriate device before making predictions.
            - The predictions are obtained by applying argmax on the model outputs.
            - The metrics are updated and computed based on the predictions and true labels.
        """
        
        model.eval()
        with torch.no_grad():
            vdata = next(iter(dataloader))
            vinputs, vlabels = vdata
            vlabels = vlabels.to(self.device).long()

            if(split!="train" and self.eval and self.eval.sliding_window):
                input_windows, label_windows = get_windows(vdata, self.eval.window, self.eval.stride, self.eval.window_target, self.eval.stride_target)
                n_vertical, n_horizontal, batch_size = input_windows.shape[:3]
                input_windows = input_windows.view(-1, *input_windows.shape[-3:])
                label_windows = label_windows.view(-1, *label_windows.shape[-2:])                    
                vdata = (input_windows, label_windows)

            preprocessed_batch = preprocess_batch(vdata, self.preprocessor)
            voutputs = get_batch_predictions_eval(preprocessed_batch, model, self.device, preprocessor=self.preprocessor)

            if(split!="train" and self.eval and self.eval.sliding_window):
                if(self.eval.window_target):
                    voutputs = voutputs.view(n_vertical, n_horizontal, batch_size, self.eval.window_target, self.eval.stride_target)     
                else:
                    voutputs = voutputs.view(n_vertical, n_horizontal, batch_size, self.eval.window, self.eval.stride)       
                voutputs = torch.cat([torch.cat(list(row), dim=-1) for row in list(voutputs)], dim=-2)

        image_counter = epoch%(5*5)//5 #Due to log_epochs being 5 and rotating 5 images
        image_counter = image_counter%len(vinputs) #In case we use a smaller batch size

        image = vinputs[image_counter].cpu().numpy()
        label = vlabels[image_counter].cpu().numpy()
        pred = voutputs[image_counter].cpu().numpy()

        return image, label, pred