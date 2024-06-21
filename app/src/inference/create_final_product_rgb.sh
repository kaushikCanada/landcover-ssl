! gdalbuildvrt mosaic.vrt ../eodata/UNET_OUTPUT/predicted/labelled/*.tif
! gdaldem color-relief mosaic.vrt color_table.txt temp_colored.tif
! gdal_translate -scale 1 8 0 255 -of PNG temp_colored.tif -outsize 30% 30% final_image.png
