def vertical_shift(img, shift_width, shift_height, shift_p):
    shift_size = random.randint(shift_width[0], shift_width[1])

    #Checking multichannel
    if len(img.shape) == 2:
        for i in range(img.shape[1] // shift_size):
            if random.random() < shift_p:
                shift = random.randint(shift_height[0], shift_height[1])
                img[shift:, i * shift_size: (i + 1) * shift_size] = img[:img.shape[0] - shift, i * shift_size: (i + 1) * shift_size]
    else:
        for i in range(img.shape[2] // shift_size):
            if random.random() < shift_p:
                shift = random.randint(shift_height[0], shift_height[1])
                img[:, shift:, i * shift_size: (i + 1) * shift_size] = img[:, :img.shape[1] - shift, i * shift_size: (i + 1) * shift_size]
    return img

def horizontal_shift(img, shift_width, shift_height, shift_p):
    shift_size = random.randint(shift_width[0], shift_width[1])

    #Checking multichannel
    if len(img.shape) == 2:
        for i in range(img.shape[0] // shift_size):
            if random.random() < shift_p:
                shift = random.randint(shift_height[0], shift_height[1])
                img[i * shift_size: (i + 1) * shift_size, shift:] = img[i * shift_size: (i + 1) * shift_size, :img.shape[1] - shift]
    else:
        for i in range(img.shape[1] // shift_size):
            if random.random() < shift_p:
                shift = random.randint(shift_height[0], shift_height[1])
                img[:, shift:, i * shift_size: (i + 1) * shift_size] = img[:, :img.shape[1] - shift, i * shift_size: (i + 1) * shift_size]

    return img


class VerticalShift(albumentations.DualTransform):
    def __init__(self, shift_width=(1, 10), shift_height=(5, 15), shift_p=0.1,  always_apply=False, p=1):
        super(VerticalShift, self).__init__(always_apply, p)
        self.shift_width = shift_width
        self.shift_height = shift_height
        self.shift_p = shift_p

    def apply(self, img, **params):
        return vertical_shift(img, shift_width=self.shift_width, shift_height=self.shift_height, shift_p=self.shift_p)

    def apply_to_mask(self, mask, **params):
        return mask

    def get_transform_init_args_names(self):
        return ()


class HorizontalShift(albumentations.DualTransform):
    def __init__(self, shift_width=(1, 10), shift_height=(5, 15), shift_p=0.1,  always_apply=False, p=1):
        super(HorizontalShift, self).__init__(always_apply, p)
        self.shift_width = shift_width
        self.shift_height = shift_height
        self.shift_p = shift_p

    def apply(self, img, **params):
        return horizontal_shift(img, shift_width=self.shift_width, shift_height=self.shift_height, shift_p=self.shift_p)

    def apply_to_mask(self, mask, **params):
        return horizontal_shift(mask, shift_width=self.shift_width, shift_height=self.shift_height, shift_p=self.shift_p)

    def get_transform_init_args_names(self):
        return ()

def invert(img):
    return 1 - img

class InvertImg(albumentations.ImageOnlyTransform):
    """Invert the input image by subtracting pixel values from 1.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        float32
    """

    def apply(self, img, **params):
        return invert(img)

    def get_transform_init_args_names(self):
        return ()