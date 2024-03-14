import zipfile
import os
import pickle
import numpy as np
import torchvision



# Create the directory for images and classes
if not os.path.exists("Imagenet64"):
    os.mkdir("Imagenet64")

# index counter
idx_ctr = 0

# Unique class set
unique_cls = set()

# Read the pickle data
dataset = torchvision.datasets.CIFAR10(root='/content/1/',download=True)
for data in dataset:
  img=data[0]
  label=data[1]
  if label==0:
    label=10
  img=np.array(img)
  img=np.transpose(img,(2,0,1))
  img=img.reshape(-1)
  img_label = dict(
    img=img,  
    label=label
    )
  unique_cls.add(label)
  with open(f"Imagenet64/{idx_ctr}.pkl", "wb") as f:
                    pickle.dump(img_label, f)

  idx_ctr += 1
            
  


# Save metadata about the number of data and
# number of classes in the data

with open(f"Imagenet64/metadata.pkl", "wb") as f:
    pickle.dump(dict(
        num_data=idx_ctr,
        cls_min=min(unique_cls),
        cls_max=max(unique_cls)
    ), f)