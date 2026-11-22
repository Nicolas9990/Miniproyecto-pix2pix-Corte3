# Miniproyecto 3 – Pix2Pix: de bosquejo a fotografía

Miniproyecto 3 del curso de Redes Neuronales y Deep Learning (UAO). Entrenamos una red generativa condicional (cGAN, arquitectura **Pix2Pix**) en **PyTorch** que recibe un bosquejo de bordes y genera la fotografía completa correspondiente.

## Integrantes
- _(nombre – usuario de GitHub)_
- _(nombre – usuario de GitHub)_
- _(nombre – usuario de GitHub)_
- _(nombre – usuario de GitHub)_
- _(nombre – usuario de GitHub)_

## Dataset
- Fotografías de paisajes de la carpeta compartida en Drive `Dataset_CycleGAN_Pintura_Fotografia` (carpetas `trainB_Fotografias`, `valB_Fotografias` y `testB_Fotografias`, ≈170 imágenes).
- El bosquejo de cada foto se genera automáticamente con detección de bordes **Canny** (trazo negro sobre fondo blanco), así que cada foto produce un par supervisado *(bosquejo, foto)*.
- Partición reproducible (semilla 42): 80 % entrenamiento, 10 % validación, 10 % prueba.

## Modelo
- **Generador:** U-Net con *skip connections* (8 niveles para 256×256) y dropout en el decoder.
- **Discriminador:** PatchGAN 70×70 que evalúa el par (bosquejo, imagen).
- **Pérdida:** cGAN (BCE) + λ·L1 con λ = 100. Adam (lr = 2e-4, β₁ = 0.5) y decaimiento lineal del *learning rate* en la segunda mitad.

## Cómo ejecutarlo (Google Colab)
1. Abrir `pix2pix_sketch2photo.ipynb` en Colab (*Archivo → Abrir cuaderno → GitHub*).
2. *Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)*.
3. En Google Drive, agregar un acceso directo de la carpeta `Dataset_CycleGAN_Pintura_Fotografia` a **Mi unidad**.
4. *Entorno de ejecución → Ejecutar todas*. El notebook monta Drive, copia las fotos, entrena, evalúa y guarda el modelo en `resultados/`.

## Validación
- Curvas de pérdida (D, G adversaria, L1 train/val).
- Métricas en el conjunto de prueba: **L1, PSNR y SSIM** (y FID opcional).
- Comparación visual bosquejo · generada · real, y prueba con dibujos hechos a mano (`datos/bocetos/`).

## Estructura
```
pix2pix_sketch2photo.ipynb   # notebook completo: datos, modelo, entrenamiento y evaluación
resultados/                  # curvas, muestras por época y métricas (se sube después de entrenar)
```

## Referencias
- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017.
- Tutorial oficial: https://www.tensorflow.org/tutorials/generative/pix2pix
