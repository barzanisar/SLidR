# # Visualization code - SLidR

# In[1]:


import os
# os.chdir('../')
import torch
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import MinkowskiEngine as ME
from datetime import datetime as dt
from torch.utils.data import DataLoader
from pretrain.model_builder import make_model
from pretrain.dataloader_nuscenes import NuScenesMatchDataset, minkunet_collate_pair_fn
from utils.transforms import make_transforms_asymmetrical_val


np.random.seed(0)


def generate_config():
    dataset = "nuscenes"
    version = "v1.0-mini"
    cylindrical_coordinates = True
    voxel_size = 0.1
    use_intensity = True
    kernel_size = 3
    model_n_out = 64
    bn_momentum = 0.05
    model_points = "minkunet"
    image_weights = "moco_v2"
    images_encoder = "resnet50"
    decoder = "dilation"
    training = "validate"
    transforms_clouds = ["Rotation", "FlipAxis"]
    transforms_mixed = ["DropCuboids", "ResizedCrop", "FlipHorizontal"]
    losses = ["loss_superpixels_average"]
    superpixels_type = "slic"
    dataset_skip_step = 1
    resume_path = "weights/minkunet_slidr_1gpu_raw.pt"

    # WARNING: DO NOT CHANGE THE FOLLOWING PARAMETERS
    # ===============================================
    if dataset.lower() == "nuscenes":
        dataset_root = "/datasets/nuscenes/"
        crop_size = (224, 416)
        crop_ratio = (14.0 / 9.0, 17.0 / 9.0)
    elif dataset.lower() == "kitti":
        dataset_root = "/datasets/semantic_kitti/"
        crop_size = (192, 672)
        crop_ratio = (3., 4.)
    else:
        raise Exception(f"Dataset Unknown: {dataset}")

    datetime = dt.today().strftime("%d%m%y-%H%M")
    
    normalize_features = True

    config = locals().copy()
    return config

config = generate_config()

mixed_transforms_val = make_transforms_asymmetrical_val(config)
dataset = NuScenesMatchDataset(
    phase="val",
    shuffle=False,
    cloud_transforms=None,
    mixed_transforms=mixed_transforms_val,
    config=config,
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=minkunet_collate_pair_fn,
    pin_memory=True,
    drop_last=False,
    worker_init_fn=lambda id:0
)
dl = iter(dataloader)


# ## Load the 2D & 3D NN

# In[2]:


model_points, model_images = make_model(config)

checkpoint = torch.load(config["resume_path"], map_location='cpu')
try:
    model_points.load_state_dict(checkpoint["model_points"])
except KeyError:
    weights = {
        k.replace("model_points.", ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model_points.")
    }
    model_points.load_state_dict(weights)

try:
    model_images.load_state_dict(checkpoint["model_images"])
except KeyError:
    weights = {
        k.replace("model_images.", ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model_images.")
    }
    model_images.load_state_dict(weights)
model_points = model_points.cuda().eval()
model_images = model_images.cuda().eval()


# ## Plotly code for dynamic figures

# In[4]:

def dynamic_heatmap(points, dist, image, save_path=None, ind=0):
    dist -= dist.min()
    dist = dist / dist.max()
    fig = go.FigureWidget(
        data=[
            dict(
                type='image',
                z=image,
                hoverinfo='skip',
                opacity=1.
            ),
            dict(
                type='scattergl',
                x=points[:, 0],
                y=points[:, 1],
                mode='markers',
                marker={'color': '#0000ff'},
                marker_size=10,
                marker_line_width=1,
                hovertemplate='<extra></extra>'
            ),
        ] +
        [dict(type='heatmap', z=dist[:,:,i], zmin=0., zmax=1., showscale=False, visible=False, hoverinfo='skip', opacity=.5) for i in range(len(points))],
    )
    fig.layout.hovermode = 'closest'
    fig.layout.xaxis.visible = False
    fig.layout.yaxis.visible = False
    fig.layout.showlegend = False
    fig.layout.width = 416
    fig.layout.height = 224
    fig.layout.plot_bgcolor="rgba(0,0,0,0)"
    fig.layout.margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0, #top margin
    )
    scatter = fig.data[1]

    # def click_fn(trace, points, selector):
    #     ind = points.point_inds[0]
    #     c = ['#0000ff'] * dist.shape[2]
    #     opacity = [0.] * dist.shape[2]
    #     c[ind] = '#ff0000'
    #     opacity[ind] = 1.
    #     if fig.data[ind + 2].visible is False:
    #         with fig.batch_update():
    #             scatter.marker.color = c
    #             scatter.marker.opacity = opacity
    #             for i in range(dist.shape[2]):
    #                 fig.data[i + 2].visible = False
    #             fig.data[ind + 2].visible = True
    #             fig.update_xaxes(range=[0., 415.])
    #             fig.update_yaxes(range=[223, 0.])
    # scatter.on_click(click_fn)

    def click_fn(ind, fig):
        c = ['#0000ff'] * dist.shape[2]
        opacity = [0.] * dist.shape[2]
        c[ind] = '#ff0000'
        opacity[ind] = 1.
        if fig.data[ind + 2].visible is False:
            with fig.batch_update():
                scatter.marker.color = c
                scatter.marker.opacity = opacity
                for i in range(dist.shape[2]):
                    fig.data[i + 2].visible = False
                fig.data[ind + 2].visible = True
                fig.update_xaxes(range=[0., 415.])
                fig.update_yaxes(range=[223, 0.])

        fig.show()
    click_fn(ind, fig)
    return fig


# In[5]:

def heatmap(points, dist, image, ind=0):
    fig = plt.figure() 
    ax = fig.add_axes([0, 0, 1, 1])
    plt.axis('off')
    ax.scatter(points[ind, 0], points[ind, 1], color='red', s=15)
    # Display the image
    ax.imshow(image/255)
    dist -= dist.min()
    dist = dist / dist.max()
    heatmap = ax.imshow(dist[:,:, ind], cmap='plasma', alpha=0.5, interpolation='none')
    plt.colorbar(heatmap)
    fig.show()

def heatmap_3d(query, points, dist_3d, image, ind=0):
    fig = plt.figure() 
    ax = fig.add_axes([0, 0, 1, 1])
    plt.axis('off')
    
    # Display the image
    ax.imshow(image/255)
    dist_3d -= dist_3d.min()
    dist_3d = dist_3d / dist_3d.max()
    heatmap = ax.scatter(points[:, 0], points[:, 1], c=dist_3d[ind], alpha=0.5, s=10, cmap='plasma')
    ax.scatter(query[ind, 0], query[ind, 1], color='red', s=15)
    # heatmap = ax.imshow(dist_3d[ind], cmap='plasma', alpha=0.5, interpolation='none')
    plt.colorbar(heatmap)
    fig.show()


def dynamic_heatmap_3d(query, points, dist_3d, image, save_path=None, ind = 0):
    dist_3d -= dist_3d.min()
    dist_3d = dist_3d / dist_3d.max()
    fig = go.FigureWidget(
        data=[
            dict(
                type='image',
                z=image,
                hoverinfo='skip'
            ),
            dict(
                type='scattergl',
                x=query[:, 0],
                y=query[:, 1],
                mode='markers',
                marker={'color': '#0000ff'},
                marker_size=10,
                marker_line_width=1,
                hovertemplate='<extra></extra>'
            ),
        ] +
        [dict(type='scatter', mode="markers", x=points[:, 0], y=points[:, 1], marker_color=dist_3d[i], marker_size=10, visible=False, hoverinfo='skip', opacity=0.5) for i in range(len(query))],
    )
    fig.layout.hovermode = 'closest'
    fig.layout.xaxis.visible = False
    fig.layout.yaxis.visible = False
    fig.layout.showlegend = False
    fig.layout.width = 416
    fig.layout.height = 224
    fig.layout.plot_bgcolor="rgba(0,0,0,0)"
    fig.layout.margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0, #top margin
    )
    scatter = fig.data[1]

    # def click_fn(trace, points, selector):
    #     ind = points.point_inds[0]
    #     c = ['#0000ff'] * dist_3d.shape[0]
    #     opacity = [0.] * dist_3d.shape[0]
    #     c[ind] = '#ff0000'
    #     opacity[ind] = 1.
    #     if fig.data[ind + 2].visible is False:
    #         with fig.batch_update():
    #             scatter.marker.color = c
    #             scatter.marker.opacity = opacity
    #             for i in range(dist_3d.shape[0]):
    #                 fig.data[i + 2].visible = False
    #             fig.data[ind + 2].visible = True
    #             fig.update_xaxes(range=[0., 415.])
    #             fig.update_yaxes(range=[223, 0.])
    # scatter.on_click(click_fn)

    def click_fn(ind, fig):
        c = ['#0000ff'] * dist_3d.shape[0]
        opacity = [0.] * dist_3d.shape[0]
        c[ind] = '#ff0000'
        opacity[ind] = 1.
        if fig.data[ind + 2].visible is False:
            with fig.batch_update():
                scatter.marker.color = c
                scatter.marker.opacity = opacity
                for i in range(dist_3d.shape[0]):
                    fig.data[i + 2].visible = False
                fig.data[ind + 2].visible = True
                fig.update_xaxes(range=[0., 415.])
                fig.update_yaxes(range=[223, 0.])
        
        fig.show()

    click_fn(ind, fig)
    return fig


# ## Process one batch

# In[6]:


with torch.no_grad():
    image_id = 0
    batch = next(dl)
    sparse_input = ME.SparseTensor(batch["sinput_F"].cuda(), batch["sinput_C"].cuda())
    output_points = model_points(sparse_input).F
    output_images = model_images(batch["input_I"].cuda())
    image = batch["input_I"][image_id].permute(1,2,0) * 255
    mask = batch["pairing_images"][:,0] == image_id
    superpixels = batch["superpixels"][image_id]
    points = np.flip(batch["pairing_images"][mask, 1:].numpy(), axis=1) #FOV points u pix, v pix in 1st cam view (N, 2)
    points_features = output_points[batch["pairing_points"][mask]] #FOV points in 1st cam view features (N, 64)
    image_features = output_images[image_id].permute(1,2,0) #(224, 416, 64)
    pairing_images = batch["pairing_images"][mask, 1:] #FOV points v pix, u pix in 1st cam view (N, 2)
    pairing_points = batch["pairing_points"][mask] #FOV points inverse index (N)
    dist_2d_3d = (1+torch.matmul(image_features, points_features.T))/2 #(224, 416, 64) x (64, N) = (224, 416, N) img to FOV pt feature similarity normalized between 0 and 1
    dist_2d_3d = dist_2d_3d.cpu().numpy()
    dist_3d_3d = (1+torch.matmul(points_features, points_features.T).cpu().numpy())/2 # FOV pt to FOV pt feature similarity normalized between 0 and 1


# ## Show the front camera for this batch

# In[7]:


fig = plt.figure(figsize=(8.32,4.48))
ax = fig.add_axes([0, 0, 1, 1])
plt.axis('off')
ax.imshow(image/255)
fig.show()


# ## Show the associated projected 3D points

# In[8]:


fig = plt.figure(figsize=(8.32,4.48))
ax = fig.add_axes([0, 0, 1, 1])
plt.axis('off')
ax.scatter(points[:, 0], points[:, 1], color='black', s=15)
ax.imshow(np.zeros((224,416,4)))


# ## Dynamic 2D features
# Clicking on a projected 3D point (in blue) will show a similarity map for the 2D features in the image relative to this point

# In[ ]:


candidates_ind = np.random.choice(points.shape[0], 10, replace=False)
# dynamic_heatmap(points[candidates_ind], dist_2d_3d[:,:,candidates_ind], image, ind = 7)
heatmap(points[candidates_ind], dist_2d_3d[:,:,candidates_ind], image, ind = 7)


# Plotting



# ## Dynamic 3D features
# Clicking on a projected 3D point (in blue) will show a similarity map for other 3D points' features, relative to this point

# In[ ]:


candidates_ind = np.random.choice(points.shape[0], 25, replace=False)
# dynamic_heatmap_3d(points[candidates_ind], points, (dist_3d_3d[candidates_ind]), image, save_path=None, ind = 0)
heatmap_3d(points[candidates_ind], points, (dist_3d_3d[candidates_ind]), image, ind = 0)



# ## PCA coloring

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(3)
y = pca.fit_transform(points_features.cpu().numpy())
y = y - y.min(0)
y = y / y.max(0)
x = pca.transform(image_features.view(-1, 64).cpu().numpy())
x = x - x.min(0)
x = x / x.max(0)
fmap = x.reshape((224,416,3))


# The following figures show a PCA coloring (in RGB) for the 2D (first figure) or 3D (second figure) features. The same PCA is used for both, so the colors are corresponding

# In[ ]:


fn = lambda x: f"rgb({x[0]}, {x[1]}, {x[2]})"
cmap = list(map(fn, (y*255).astype(np.int32)))
fig = go.FigureWidget(
    data=[
        dict(
            type='image',
            z=image,
            hoverinfo='skip'
        ),
        dict(
            type='image',
            z=fmap*255,
            hoverinfo='skip',
            opacity=0.5
        )
    ]
)
fig.layout.xaxis.visible = False
fig.layout.yaxis.visible = False
fig.layout.showlegend = False
fig.layout.width = 416
fig.layout.height = 224
fig.layout.plot_bgcolor="rgba(0,0,0,0)"
fig.layout.margin=go.layout.Margin(
    l=0, #left margin
    r=0, #right margin
    b=0, #bottom margin
    t=0, #top margin
)
fig.update_xaxes(range=[0., 415.])
fig.update_yaxes(range=[223, 0.])
fig


# In[ ]:


fn = lambda x: f"rgb({x[0]}, {x[1]}, {x[2]})"
cmap = list(map(fn, (y*255).astype(np.int32)))
fig = go.FigureWidget(
    data=[
        dict(
            type='image',
            z=image,
            hoverinfo='skip'
        ),
        dict(type='scatter', mode="markers", x=points[:, 0], y=points[:, 1], marker_color=cmap, marker_size=10, visible=True, hoverinfo='skip', opacity=0.5)
    ]
)
fig.layout.xaxis.visible = False
fig.layout.yaxis.visible = False
fig.layout.showlegend = False
fig.layout.width = 416
fig.layout.height = 224
fig.layout.plot_bgcolor="rgba(0,0,0,0)"
fig.layout.margin=go.layout.Margin(
    l=0, #left margin
    r=0, #right margin
    b=0, #bottom margin
    t=0, #top margin
)
fig.update_xaxes(range=[0., 415.])
fig.update_yaxes(range=[223, 0.])
fig


# In[ ]:


fig = plt.figure(figsize=(8.32,4.48))
ax = fig.add_axes([0, 0, 1, 1])
plt.axis('off')
ax.scatter(points[:, 0], points[:, 1], color=y, s=15)
ax.imshow(np.zeros((224,416,4)))
fig.show()


# ## Pooling the PCA coloring by superpixels

# In[ ]:


m = tuple(pairing_images.cpu().T.long())

superpixels_I = superpixels.flatten()
idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
total_pixels = superpixels_I.shape[0]
idx_I = torch.arange(total_pixels, device=superpixels.device)

one_hot_P = torch.sparse_coo_tensor(
    torch.stack((
        superpixels[m], idx_P
    ), dim=0),
    torch.ones(pairing_points.shape[0], device=superpixels.device),
    (superpixels.max() + 1, pairing_points.shape[0])
)

one_hot_I = torch.sparse_coo_tensor(
    torch.stack((
        superpixels_I, idx_I
    ), dim=0),
    torch.ones(total_pixels, device=superpixels.device),
    (superpixels.max() + 1, total_pixels)
)

k = one_hot_P @ points_features.cpu()
k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
k = pca.transform(k.cpu().numpy())
k = k - k.min(0)
k = k / k.max(0) #(num segments in 1st cam view, 3 PCA components from segment-wise point backbone feats) -> each col max = 1 and col min = 0 
q = one_hot_I @ image_features.cpu().flatten(0, 1)
q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)
q = pca.transform(q.cpu().numpy())
q = q - q.min(0)
q = q / q.max(0) #(num segments in 1st cam view, 3 PCA components from segment-wise image backbone feats) -> each col max = 1 and col min = 0 


# In[ ]:


fig = plt.figure(figsize=(8.32,4.48))
ax = fig.add_axes([0, 0, 1, 1])
plt.axis('off')
ax.imshow(q[superpixels.numpy()])# segment features from image backbone (pca'd version) displayed on image superpixels
fig.show()


# In[ ]:


fig = plt.figure(figsize=(8.32,4.48))
ax = fig.add_axes([0, 0, 1, 1])
plt.axis('off')
ax.scatter(points[:, 0], points[:, 1], color=k[superpixels[m]], s=15)
ax.imshow(np.zeros((224,416,4))) #segment feature from point backbone shown on each FOV pt as a pca
fig.show()


# ## Showing superpixels

# In[ ]:


scene_index = 0 #np.random.randint(850)
current_sample_token = dataloader.dataset.nusc.scene[scene_index]['first_sample_token']
data = dataloader.dataset.nusc.get("sample", current_sample_token)['data']
cam_info = dataloader.dataset.nusc.get("sample_data", data['CAM_FRONT_RIGHT'])
token = cam_info['token']
filename = cam_info['filename']
im = plt.imread(f"datasets/nuscenes/v1.0-mini/{filename}")
fig = plt.figure(figsize=(8,4.5))
ax = fig.add_axes([0, 0, 1, 1])
plt.axis('off')
plt.imshow(im)


# In[ ]:


from PIL import Image
sp = np.array(Image.open(f"superpixels/nuscenes/superpixels_slic/{token}.png"))
from skimage.segmentation import mark_boundaries
compound_image = np.zeros((900,1600,3))
for i in range(sp.max()):
    ma = sp==i
    compound_image[ma] = np.average(im[ma], 0) / 255
compound_image = mark_boundaries(compound_image, sp, color=(1., 1., 1.))
fig = plt.figure(figsize=(8,4.5))
ax = fig.add_axes([0, 0, 1, 1])
plt.axis('off')
plt.imshow(compound_image)


# In[ ]:


fig = plt.figure(figsize=(8.32,4.48))
ax = fig.add_axes([0, 0, 1, 1])
plt.axis('off')
ax.scatter(points[:, 0], points[:, 1], color=np.array(im[list(np.flip(points, 1).T)] / 255), s=15) #color for each FOV pt = img[v pix, u pix]
ax.imshow(np.zeros((224,416,4)))
fig.show()
b=1

