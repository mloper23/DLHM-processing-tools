from pyLHM.myfunctions import *
import plotly.express as px

rec = reconstruct()

path = '/Users/mjloperaa/Library/CloudStorage/OneDrive-UniversidadEAFIT/EAFIT/DLHM-data/05242023/usaf6.tif'
img = np.array(cv.imread(path))[:, 200:1224, 1].astype('float')

N, M = img.shape

FC = rec.filtcosenoF(100, N, 0)

K = rec.kreuzer3F(0.68e-3, img, 532e-9, 3.6e-6, 3.06e-7, 8e-3, FC)

fig = px.imshow(np.angle(K))
fig.write_html('test.html')
