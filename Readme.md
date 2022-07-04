### **Healthcare Demo built using FastAPI**<br>

- Uses `python-3.8.10`
- API built using FastAPI
- Locally hosted at `http://127.0.0.1:10000`
- Functionalities Present:
    - Diabetes Prediction `/infer/diabetes`<br>
    - Cardiovascular Disease Prediction `/infer/cardiovascular-disease`<br>
    - Brain MRI Abnormality Segmentation `/infer/brain-mri`<br>
    - Pneumonia Prediction using Chest X-Rays `/infer/pneumonia`<br>
    - Tuberculosis Prediction using Chest X-Rays `/infer/tuberculosis`<br>

<br>

### **Detailed Information**

<br>

1. Install Python
2. Run `pip install virtualenv`
3. Run `make-env.bat` or `make-env-3.9.bat`
4. Run `start-api-server.bat`. The API will now be served at `http://127.0.0.1:10000` (Alternatively, setup `.vscode`)

<br>

### **Notes on Models**

<br>

<pre>
1. brain-abnormality : pretrained
2. diabetes          : gbc
3, cardiovascular    : abc
4. pneumonia         : d121-na384-oclr-f2-0.5
5. tuberculosis      : d121-na384-oclr-f4
</pre>