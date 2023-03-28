import numpy as np
import random
import cv2
from PIL import Image
import imgaug as ia
import math
from imgaug import augmenters as iaa
from augmentation.multi_stage_blur import multi_stage_blur

class ImageAugmentation:
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def augment(self, image, input_channel):
        aug_result = ''

        image = np.array(image, dtype=np.uint8)
        if input_channel == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # アスペクト比変更
        if random.randint(0,1):
            ratio = random.uniform(0.7, 1.3)
            height, width = image.shape[:2]
            image = cv2.resize(image, (int(width*ratio), height))
            aug_result += f'aspect{ratio:.1f} - '
        # 角度
        if random.random() > 0.8: 
            angle = random.uniform(-5, 5)
            image = self.rotate_image(image, angle)
            aug_result += f'angle{angle:.1f} - '
        
        aug_p = random.random()
        if aug_p > 0.5:
            # 基本的な拡張
            if random.randint(0, 1):
                seq = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.OneOf([
                            iaa.AdditiveGaussianNoise(scale=0.10*1),
                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 15)),
                            iaa.Salt(0.2),
                            iaa.Lambda(self.aug_salt),
                            iaa.AdditivePoissonNoise(lam=(0.0, 15.0)),
                            iaa.AdditiveLaplaceNoise(scale=(0, 15)),
                            iaa.SaltAndPepper(0.03, 0.1)
                            ])),
                iaa.Sometimes(0.5, iaa.OneOf([
                            iaa.GaussianBlur((0, 1.0)),
                            iaa.AverageBlur(k=(2, 3)),
                            iaa.Lambda(self.aug_random_blur),
                            iaa.Lambda(self.aug_downsample),
                            iaa.JpegCompression(compression=(0, 40))
                        ]))
                ])
                image = seq(image=image)
                aug_result += f'basic_aug - '
            # 強めのノイズ
            else:
                seq = iaa.Sequential([
                    iaa.SomeOf((0, 4),[
                     iaa.OneOf([
                          iaa.Add((-150, 150)),
                          iaa.Multiply((0.5, 1.5)),
                          iaa.Multiply((0.5, 1.5), per_channel=0.5),
                          iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                         ]),
                     iaa.ChannelShuffle(0.35, channels=[0, 1]),
                     iaa.Sharpen(alpha=0.5),
                     iaa.OneOf([
                         iaa.AdditiveGaussianNoise(scale=0.1*255),
                         # iaa.Rain(speed=(0.1,0.3))
                     ]),
                    iaa.ElasticTransformation(alpha=(0.9, 1.1), sigma=0.25)
                    ])]
                )
                image = seq(image=image)
                aug_result += f'heavy_aug - '
        elif 0.5 > aug_p > 0.2:
            image = self.aug_multi_stage_blur(image)
            aug_result += f'multi_stage - '
        elif 0.2 > aug_p > 0.1: # downscale
            image = iaa.Lambda(self.aug_downsample)(image=image)
            aug_result += f'downscale - '
        if random.random() < 0.1: # 白字対応Bit反転
           image = ~image
           aug_result += f'bit_inversion - '
        if random.random() < 0.1: # MotionBlur
           image = iaa.MotionBlur()(image=image)
           aug_result += f'MotionBlur - '

        # PILImageに変換
        if input_channel == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.verbose:
            print(aug_result)
        return image
    
    def rotate_image(self, image, angle):
        height, width = image.shape[:2]
        angle_rad = math.radians(angle)

        # 回転後の画像サイズを計算
        w_rot = int(np.round(height*np.absolute(np.sin(angle_rad))+width*np.absolute(np.cos(angle_rad))))
        h_rot = int(np.round(height*np.absolute(np.cos(angle_rad))+width*np.absolute(np.sin(angle_rad))))
        size_rot = (w_rot, h_rot)

        # 元画像の中心を軸に回転する
        center = (width/2, height/2)
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # 平行移動を加える (rotation + translation)
        affine_matrix = rotation_matrix.copy()
        affine_matrix[0][2] = affine_matrix[0][2] -width/2 + w_rot/2
        affine_matrix[1][2] = affine_matrix[1][2] -height/2 + h_rot/2

        image = cv2.warpAffine(image, affine_matrix, size_rot, flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        return image

    def aug_salt(self, images, random_state, parents, hooks):
        for image in images:
            prob = random.uniform(0.05, 0.15)
            thres = 1 - prob 
            if len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
                color = 255
            else:
                color = (255, 255, 255)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    rdn = random.random()
                    if rdn > thres: # salt noise
                        image[i][j] = color
        return images

    def aug_downsample(self, images, random_state, parents, hooks):
        new_images = []
        interpolations1 = [cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]
        interpolations2 = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

        for image in images:
            if len(image.shape) == 3:
                height, width, channel = image.shape
            else:
                height, width = image.shape
                channel = 1
            if height <= 0 or width <= 0:
                print(image.shape)
                new_images.append(image)
                continue
            image = cv2.resize(image, (width//2, height//2), interpolation=random.choice(interpolations1))
            image = cv2.resize(image, (width, height), interpolation=random.choice(interpolations2))
            if channel == 1:
                image = image.reshape((height, width, channel))
            new_images.append(image)
        return new_images
    
    def aug_random_blur(self, images, random_state, parents, hooks):
        new_images = []
        for image in images:
            if len(image.shape) == 3:
                height, width, channel = image.shape
            else:
                height, width = image.shape
                channel = 1
            image = multi_stage_blur(image)
            if channel == 1:
                image = image.reshape((height, width, channel))
            new_images.append(image)
        return new_images
    
    def aug_multi_stage_blur(self, image):
        return multi_stage_blur(image)