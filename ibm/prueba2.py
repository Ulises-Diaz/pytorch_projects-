import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Cargar la nube de puntos de ejemplo
dataset = o3d.data.OfficePointClouds()
# Select just the first point cloud for this example
pcd = o3d.io.read_point_cloud(dataset.paths[0])
pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd)

# Visualización 3D de la nube de puntos
print("Visualizando nube de puntos en 3D...")
o3d.visualization.draw_geometries([pcd])

# Para convertir a imagen, necesitamos convertir primero al formato tensor
pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd)

# Definir parámetros de cámara
width = 640
height = 480
fx = 600.0
fy = 600.0
cx = width / 2
cy = height / 2

# Crear matriz de intrínsecos
intrinsics = o3d.core.Tensor([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=o3d.core.Dtype.Float64)

# Proyectar a imagen RGBD
rgbd_image = pcd_t.project_to_rgbd_image(
    width=width,
    height=height,
    intrinsics=intrinsics,
    depth_scale=1000.0,
    depth_max=5.0
)

# Visualizar las imágenes resultantes con matplotlib
plt.figure(figsize=(12, 6))

# Imagen RGB - corrección aquí
plt.subplot(1, 2, 1)
plt.title("RGB Image")
# Usamos .as_tensor().numpy() en lugar de .cpu().numpy()
rgb_img = np.asarray(rgbd_image.color.to_legacy())
plt.imshow(rgb_img)

# Imagen de profundidad - corrección aquí
plt.subplot(1, 2, 2)
plt.title("Depth Image")
depth_img = np.asarray(rgbd_image.depth.to_legacy())
plt.imshow(depth_img, cmap='jet')

plt.tight_layout()
plt.show()

# Guardar las imágenes
o3d.io.write_image("rgb_image.png", rgbd_image.color.to_legacy())
o3d.io.write_image("depth_image.png", rgbd_image.depth.to_legacy())

print("Imágenes guardadas correctamente.")