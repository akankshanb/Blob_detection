# Blob_detection

The objective of this project is to implement a Laplacian blob detector by implementing the entire algorithm independently without built-in functions used for core components.

<head><b>Algorithm outline:</b></head>

• Generate a Laplacian of Gaussian filter.</br>
• Build a Laplacian scale space, starting with some initial scale and going for n iterations:</br>
  Filter image with scale-normalized Laplacian at current scale.</br>
  Save square of Laplacian response for current level of scale space. o Increase scale by a factor 𝑘.</br>
• Perform non-maximum suppression in scale space.</br>
• Display resulting circles at their characteristic scales.</br>


<head><b>Libraries used:</b></head>
cv2</br>
numpy</br>
math</br>
time</br>
