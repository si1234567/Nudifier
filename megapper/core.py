import io
import time
import os
import math
import cv2
import onnxruntime
from PIL import Image
import numpy as np
from typing import Optional, Union, List, Tuple


class upscale(object):
    def __init__(self, 
                 model: str = 'waifu2x_art',
                 scale: Optional[float] = None,
                 size: Optional[Union[int, List[int], Tuple[int]]] = None,
                 denoise_level: int =  2, # [-1, 3], -1 means no denoise
                 use_gpu: bool = False, 
                 device_id: int = 0,
                 ocr_text: bool = False,
                 ocr_font_size: int = 28,
                 ocr_font_color: Union[Tuple[int], List[int]] = (0, 0, 0),
                 ocr_background_color: Union[Tuple[int], List[int]] = (255, 255, 255),
                 ocr_font_ttf: Optional[str] = None,
                 verbose: bool = False,
                ):

        self.scale = scale
        self.size = size
        self.model = model
        self.denoise_level = denoise_level
        self.ocr_text = ocr_text
        self.ocr_font_size = ocr_font_size
        self.ocr_font_color = ocr_font_color
        self.ocr_background_color = ocr_background_color
        self.ocr_font_ttf = ocr_font_ttf
        self.verbose = verbose

        if model == 'waifu2x_art':
            if denoise_level < 0:
                model_path = f'cunet/scale2.0x_model.onnx'
            else:
                model_path = f'cunet/noise{denoise_level}_scale2.0x_model.onnx'
        elif model == 'waifu2x_photo':
            if denoise_level < 0:
                model_path = f'upconv/scale2.0x_model.onnx'
            else:
                model_path = f'upconv/noise{denoise_level}_scale2.0x_model.onnx'
        else:
            raise NotImplementedError

        self.__model_path = os.path.join(os.path.dirname(__file__), 'models', model_path)

        if use_gpu:
            if verbose:
                print(f'[INFO] use GPU {device_id}')
            self.__providers = [
                ('CUDAExecutionProvider', {
                    'device_id': device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cuda_mem_limit': 4 * 1024 * 1024 * 1024, # 4GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
            ]
        else:
            self.__providers = [
                'CPUExecutionProvider',
            ]

        self.__ort_session = onnxruntime.InferenceSession(self.__model_path, providers=self.__providers)
        self.__ort_input_name = self.__ort_session.get_inputs()[0].name

    def __call__(self, 
            image: Union[np.ndarray, str],
            output_path: Optional[str] = None,
            window_size: int = 256,
           ):
        return self.run(image, output_path, window_size)

    def run(self, 
            image: Union[np.ndarray, str],
            output_path: Optional[str] = None,
            window_size: int = 256,
           ):
        
        # image: [H, W, 3] or [H, W, 4] or path
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

        if self.verbose:
            print(f'[INFO] input: {image.shape} {image.dtype}')

        # determine out size
        in_size = np.array(image.shape[:2])
        if self.scale is not None:
            out_size = (in_size * self.scale).astype(int)
        elif self.size is not None:
            if isinstance(self.size, int):
                out_size = np.array([self.size, self.size])
            else:
                out_size = np.array(self.size)
        else:
            # default is 2x
            out_size = in_size * 2

        if self.verbose:
            print(f'[INFO] output: {out_size}')

        # determine run times
        iter_2x = math.ceil(np.log2(out_size / in_size).max())

        # process image
        rgb = image[:, :, :3]
        alpha = image[:, :, 3] if image.shape[2] == 4 else None
        
        rgb = rgb.transpose(2,0,1).astype(np.float32) / 255

        # run
        if self.model == 'waifu2x_art':
            padding = 18
        elif self.model == 'waifu2x_photo':
            padding = 7
        else:
            padding = 0

        x = np.expand_dims(rgb, axis=0) # [1, 3, H, W]

        while iter_2x:
            
            if self.verbose:
                print(f'[INFO] 2x scaling remaining: {iter_2x}, window size = {window_size}')

            if window_size == -1:
                h, w = x.shape[2], x.shape[3]
                extra_padding_h = 1 if h % 2 != 0 else 0
                extra_padding_w = 1 if w % 2 != 0 else 0
                x = np.pad(x, ((0, 0), (0, 0), (padding, padding + extra_padding_h), (padding, padding + extra_padding_w)))
                x = self.__ort_session.run(None, {self.__ort_input_name: x})[0]
                x = x[:, :, :2*h, :2*w]

            else:
                h, w = x.shape[2], x.shape[3]
                pad_x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding))).astype(np.float32)
                y =  np.zeros((1, 3, h * 2, w * 2), dtype=np.float32)
                cnt = np.zeros((h * 2, w * 2))
                for h0 in range(0, h, window_size):
                    for w0 in range(0, w, window_size):
                        h1 = min(h, h0 + window_size)
                        w1 = min(w, w0 + window_size)
                        h0 = min(h0, h1 - window_size)
                        w0 = min(w0, w1 - window_size)
                        if self.verbose:
                            print(f'[INFO] process window {h0: >5}:{h1: >5} / {w0: >5}:{w1: >5}', end='\r')
                        patch_x = pad_x[:, :, h0:h1+padding*2, w0:w1+padding*2]
                        patch_y = self.__ort_session.run(None, {self.__ort_input_name: patch_x})[0]
                        y[:, :, h0*2:h1*2, w0*2:w1*2] += patch_y
                        cnt[h0*2:h1*2, w0*2:w1*2] += 1
                x = y / cnt

            iter_2x -= 1

        result = (x[0].clip(0, 1) * 255).astype(np.uint8).transpose(1,2,0) # [3, H', W'] --> [H', W', 3]

        # alpha composition
        if alpha is not None:
            alpha = cv2.resize(alpha, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_CUBIC)
            result = np.concatenate([result, np.expand_dims(alpha, -1)], axis=-1)

        # adjust for non-propotional scaling
        if result.shape[0] != out_size[0] or result.shape[1] != out_size[1]:
            result = cv2.resize(result, (out_size[1], out_size[0]), interpolation=cv2.INTER_CUBIC)

        # write 
        if output_path is not None:
            if self.verbose:
                print(f'[INFO] write: {output_path}')
            cv2.imwrite(output_path, result)

        return result
