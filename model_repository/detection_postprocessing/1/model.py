# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import io
import json
import sys
import time
import numpy as np
from PIL import Image
import logging
import copy
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

from postprocess import (DBPostProcess, 
                         sorted_boxes, 
                         get_rotate_crop_image, 
                         get_minarea_rect_crop_poly, 
                         recognition_resize_norm_img,
                         filter_tag_det_res_only_clip,
                         filter_tag_det_res
                         )



class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "detection_postprocessing_output"
        )

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )


    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype
        DET_BOX_TYPE = 'quad'

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            
            _featuremap = pb_utils.get_input_tensor_by_name(
                request, "detection_postprocessing_input"
            )

            _shape_list = pb_utils.get_input_tensor_by_name(
                request, "original_image_info"
            )

            original_img_stream = pb_utils.get_input_tensor_by_name(
                request, "original_image"
            )

            featuremap = _featuremap.as_numpy()
            shape_list = _shape_list.as_numpy()

            original_img = original_img_stream.as_numpy()
            original_img = np.array(Image.open(io.BytesIO(original_img.tobytes())))
            ori_im = original_img.copy()

            logging.debug('initialize postprocessing class DBPostProcess')
            # get detection boxes from heatmap
            postprocess_op = DBPostProcess(box_type = DET_BOX_TYPE)
            logging.debug(f'featuremap : {featuremap.shape}')
            logging.debug(f'shape_list : {shape_list}')
            post_result = postprocess_op(featuremap, shape_list)
            logging.debug('post result : ', post_result)
            dt_boxes = post_result[0]['points']


            if DET_BOX_TYPE == 'poly':
                dt_boxes = filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
            else:
                dt_boxes = filter_tag_det_res(dt_boxes, ori_im.shape)

            dt_boxes = sorted_boxes(dt_boxes)

            # cropped_images
            img_crop_list = []
            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                if DET_BOX_TYPE == "quad":
                    img_crop = get_rotate_crop_image(ori_im, tmp_box)
                else:
                    # img_crop = get_minarea_rect_crop(ori_im, tmp_box)
                    # tick1 = time.time()
                    img_crop = get_minarea_rect_crop_poly(ori_im, tmp_box)
                    # print("time for get_minarea_rect_crop_poly: {}".format(time.time() - tick1))



                # resize img_crop
                # TODO : batch 나누기, max_wh_ratio
                resized_norm_img_crop = recognition_resize_norm_img(img_crop, max_wh_ratio = None) 
                resized_norm_img_crop = resized_norm_img_crop[np.newaxis, :]
                img_crop_list.append(resized_norm_img_crop)
            
            img_crop_batch = np.concatenate(img_crop_list)
            img_crop_batch = img_crop_batch.copy()

            logging.debug(dt_boxes)
            

            
            out_tensor_0 = pb_utils.Tensor(
                "detection_postprocessing_output", img_crop_batch.astype(output0_dtype)
            )

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)
        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")

