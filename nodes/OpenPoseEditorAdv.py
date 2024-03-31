import torch
import json
from nodes import LoadImage


class OpenPoseEditorAdv:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "image": ("STRING", { "default": "" }),
                    "savedPose": ("STRING", { "default": "" })
                }
            }

    RETURN_TYPES = ("IMAGE","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK")
    FUNCTION = "load_image"

    CATEGORY = "image"

    def load_image(self, image, savedPose):
        image, mask = LoadImage.load_image(self, image)
        savedPose = json.loads(savedPose)
        # print(savedPose['width'], savedPose['height'])

        width = savedPose['width']
        height = savedPose['height']
        retVal = [ image ]
        for pose in savedPose['keypoints']:
            maxCoordinates = [0,0]
            minCoordinates = [9999,9999]
            for point in pose:
                if point[0] > maxCoordinates[0]: maxCoordinates[0] = min(int(point[0]), width)
                if point[1] > maxCoordinates[1]: maxCoordinates[1] = min(int(point[1]), height)
                if point[0] < minCoordinates[0]: minCoordinates[0] = max(int(point[0]), 0)
                if point[1] < minCoordinates[1]: minCoordinates[1] = max(int(point[1]), 0)
            # print(minCoordinates, maxCoordinates)
            mask = torch.full((savedPose['height'], savedPose['width']), 0.0, dtype=torch.float32, device="cpu")
            mask[minCoordinates[1]:maxCoordinates[1], minCoordinates[0]:maxCoordinates[0]] = 1.0
            retVal.append(mask)

        return tuple(retVal)
