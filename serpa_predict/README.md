Serpa Predict (Flutter)

Resumo:
- App offline Android com inferencia TFLite (multimodal)
- Entradas: manual e CSV
- Saidas: pH, OD, TURB + indice de poluicao
- Mapa real via OpenStreetMap (requer internet para tiles, a menos que use tiles locais)

Arquivos esperados:
- assets/model/model.tflite
- assets/model/norm_multi.json

Passos:
1) Gerar TFLite
   source /home/alfredo/miniconda3/etc/profile.d/conda.sh
   conda activate export_tflite
   cd ../scripts_multimodal
   python export_tflite.py

2) Copiar para o app
   cp outputs/model.tflite ../serpa_predict/assets/model/model.tflite
   # se o arquivo nao aparecer acima, use:
   # cp outputs/onnx2tf/multimodal_float32.tflite ../serpa_predict/assets/model/model.tflite

3) Copiar MBTiles para o Android (offline)
   adb push "/home/alfredo/Downloads/osm-2020-02-10-v3.11_south-america_brazil_t.mbtiles" /sdcard/Download/

4) Rodar no Android
   cd ../serpa_predict
   flutter pub get
   flutter run
