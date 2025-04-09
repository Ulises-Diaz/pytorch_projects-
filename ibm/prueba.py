import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_rgbd_pytorch(rgb_tensor, depth_tensor):
    plt.figure(figsize=(12, 6))
    
    # RGB
    plt.subplot(1, 2, 1)
    plt.title("RGB Imag")
    # Convertir tensor PyTorch a NumPy para visualización
    rgb_img = rgb_tensor.permute(1, 2, 0).numpy() 
    plt.imshow(rgb_img)
    
    # DEPTH
    plt.subplot(1, 2, 2)
    plt.title("Depth Image")
    depth_img = depth_tensor.squeeze().numpy()  # Eliminar dimensión de canal si existe
    plt.imshow(depth_img, cmap='jet')
    
    plt.show()

# 1. Cargar una nube de puntos. LO SAQUE DE UN DATASET PRUEBA. HABRIA QUE USAR UN PCD DEL DATASET O ALGO. UN FOR PARA QUE CONVIERTA TODAS A PCD
dataset = o3d.data.OfficePointClouds()
pcd = o3d.io.read_point_cloud(dataset.paths[0])
print("Nube de puntos cargada:", pcd)

pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd)

# 2. Param imagen 
width = 1280 # Resolucion
height = 720 # Resolucion 
fx = 200 # Distancia Focal. entre mas pequeno, angulo de vision mayor
fy = 200 # Distancia Focal x2 . CAMBIAR ESTOS PARAM PARA TENER BIEN LA FOTO . AMPLITUD CAM
cx = 100  # centro de la imagen x
cy = 100  # centro de la imagen y



# 3. Crear tensor de intrínsecos
intrinsics = o3d.core.Tensor([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=o3d.core.Dtype.Float64)

# 4. Definir extrínsecos
extrinsics = o3d.core.Tensor.eye(4, dtype=o3d.core.Dtype.Float64)


# 5. Proyectar la nube de puntos a imagen RGBD. FUNCION IMPORTANTE
rgbd_image = pcd_t.project_to_rgbd_image(
    width=width,
    height=height,
    intrinsics=intrinsics,
    extrinsics=extrinsics,
    depth_scale=1000.0,
    depth_max=3.0
)

# 6. Convertir las imágenes de Open3D a arrays NumPy
rgb_np = np.asarray(rgbd_image.color.to_legacy())
depth_np = np.asarray(rgbd_image.depth.to_legacy())

# 7. Convertir NumPy arrays a tensores PyTorch
rgb_torch = torch.from_numpy(rgb_np).float()
depth_torch = torch.from_numpy(depth_np).float()

# Reorganizar RGB de (H,W,C) a (C,H,W) como es común en PyTorch
if len(rgb_torch.shape) == 3:  # Si tiene canales
    rgb_torch = rgb_torch.permute(2, 0, 1)

# Añadir dimensión de canal si depth no lo tiene
if len(depth_torch.shape) == 2:
    depth_torch = depth_torch.unsqueeze(0)

print(f"Forma del tensor PyTorch RGB: {rgb_torch.shape}")
print(f"Forma del tensor PyTorch de profundidad: {depth_torch.shape}")
print(f"Tipo de datos RGB: {rgb_torch.dtype}")
print(f"Tipo de datos profundidad: {depth_torch.dtype}")
print(f"Dispositivo: {rgb_torch.device}")

# 8. Visualizar usando los tensores PyTorch
print("Visualizando imagen RGBD (PyTorch)...")
visualize_rgbd_pytorch(rgb_torch, depth_torch)

# GPU
if torch.cuda.is_available():
    rgb_torch = rgb_torch.cuda()
    depth_torch = depth_torch.cuda()
    print("Tensores movidos a GPU")
    print(f"Dispositivo: {rgb_torch.device}")

# Guardar imagenes
print("Images guardadas")

# Convertir tensores PyTorch de nuevo a formato que Open3D pueda guardar
rgb_save = rgb_torch.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
depth_save = depth_torch.squeeze().cpu().numpy()

# Crear imágenes Open3D a partir de los arrays NumPy
rgb_o3d = o3d.geometry.Image(rgb_save)
depth_o3d = o3d.geometry.Image(depth_save)

