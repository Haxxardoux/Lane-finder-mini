import torch
import numpy as np
import torch.nn.functional as F
import cv2

class vidSet(torch.utils.data.Dataset):
    def __init__(self, video_paths):
<<<<<<< HEAD
=======

>>>>>>> 68158a80eb37179d249edf230538b9f647159d7d
        self.video_paths = video_paths
        self.caps = [cv2.VideoCapture(video_path) for video_path in self.video_paths]
        self.images = np.array([[capid, framenum] for capid, cap in enumerate(self.caps) for framenum in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))])[0::20]
        self.images2 = np.array([[capid, framenum] for capid, cap in enumerate(self.caps) for framenum in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))])[1::20]
        
<<<<<<< HEAD
        if self.images.shape != self.images2.shape:
#             print(self.images.shape)
=======
        if self.images.shape != self.images.shape:
>>>>>>> 68158a80eb37179d249edf230538b9f647159d7d
            self.images = self.images[:-1]
        
        self.ret = np.hstack((self.images, self.images2))
        self.ret = self.ret.reshape(self.ret.shape[0]*2, 2)

    def __len__(self):
         return len(self.ret)

    def __getitem__(self, idx):
        capid, framenum = self.ret[idx]
        cap = self.caps[capid]
        cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
        res, frame = cap.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
        img_tensor = F.interpolate(img_tensor, scale_factor=(0.4, 0.4))

        return img_tensor.squeeze()

