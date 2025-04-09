import numpy as np
import cv2

def gini(x):
    """
    Calcula el coeficiente Gini de un array numpy.
    
    Args:
        x: Array numpy de valores
        
    Returns:
        Coeficiente Gini
    """
    # Asegurarse de que los datos son un array numpy
    x = np.array(x, dtype=np.float32)
    
    # Eliminar valores negativos
    x = np.abs(x)
    
    # Ordenar valores
    x = np.sort(x)
    
    # Índices desde 1 hasta n
    n = len(x)
    index = np.arange(1, n + 1)
    
    # Calcular coeficiente Gini
    return ((2 * np.sum(x * index)) / (n * np.sum(x))) - ((n + 1) / n)

def load_and_preprocess_image(image_path = '', size=(256, 256)):
    """
    Carga y preprocesa una imagen usando OpenCV.
    
    Args:
        image_path: Ruta a la imagen
        size: Tupla con el tamaño deseado de la imagen (ancho, alto)
    
    Returns:
        Imagen preprocesada en escala de grises
    """
    # Leer la imagen
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
    # Convertir a escala de grises si la imagen es a color
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Redimensionar la imagen
    img = cv2.resize(img, size)
    
    # Normalizar valores a rango [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def calculate_roughness_metric(image, reference_roughness_mean=None, reference_roughness_std=None):
    """
    Calcula la métrica de Roughness (MSR) usando OpenCV.
    
    Args:
        image: Imagen de entrada (numpy array en escala de grises)
        reference_roughness_mean: Media pre-calculada del conjunto de referencia
        reference_roughness_std: Desviación estándar pre-calculada del conjunto de referencia
    
    Returns:
        MSR score o R_image dependiendo de si se proporcionan estadísticas de referencia
    """
    # Asegurarse de que la imagen esté en float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # Calcular el Laplaciano usando OpenCV
    laplacian = cv2.Laplacian(image, cv2.CV_32F)
    
    # Crear histograma de 30 bins del Laplaciano
    hist = cv2.calcHist([laplacian], [0], None, [30], 
                        [laplacian.min(), laplacian.max()])
    
    # Normalizar el histograma
    hist = hist.ravel() / hist.sum()
    
    # Calcular el coeficiente Gini del histograma
    R_image = gini(hist)
    
    # Si no se proporcionan estadísticas de referencia, devolver solo R_image
    if reference_roughness_mean is None or reference_roughness_std is None:
        return R_image
    
    # Calcular MSR usando la ecuación del paper
    MSR = np.abs(R_image - reference_roughness_mean) / reference_roughness_std
    
    return MSR

def calculate_reference_statistics(reference_image_paths, size=(256, 256)):
    """
    Calcula las estadísticas de referencia para un conjunto de imágenes.
    
    Args:
        reference_image_paths: Lista de rutas a las imágenes de referencia
        size: Tupla con el tamaño deseado de las imágenes
    
    Returns:
        tuple: (media de roughness, desviación estándar de roughness)
    """
    roughness_values = []
    
    for path in reference_image_paths:
        try:
            # Cargar y preprocesar imagen
            image = load_and_preprocess_image(path, size)
            # Calcular roughness
            R_image = calculate_roughness_metric(image)
            roughness_values.append(R_image)
        except Exception as e:
            print(f"Error procesando {path}: {str(e)}")
            continue
    
    if not roughness_values:
        raise ValueError("No se pudo procesar ninguna imagen de referencia")
    
    return np.mean(roughness_values), np.std(roughness_values)

# Ejemplo de uso
def main():
    # Rutas de ejemplo
    test_image_path = "ruta/a/tu/imagen_test.jpg"
    reference_paths = ["ruta/a/referencia1.jpg", "ruta/a/referencia2.jpg"]
    
    try:
        # Calcular estadísticas de referencia
        print("Calculando estadísticas de referencia...")
        ref_mean, ref_std = calculate_reference_statistics(reference_paths)
        print(f"Media de referencia: {ref_mean:.4f}")
        print(f"Desviación estándar de referencia: {ref_std:.4f}")
        
        # Procesar imagen de prueba
        print("\nProcesando imagen de prueba...")
        test_image = load_and_preprocess_image(test_image_path)
        msr = calculate_roughness_metric(test_image, ref_mean, ref_std)
        print(f"MSR Score: {msr:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()