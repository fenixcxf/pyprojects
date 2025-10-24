import pims


@pims.pipeline
def gray(image):
    return image[:, :, 1]
