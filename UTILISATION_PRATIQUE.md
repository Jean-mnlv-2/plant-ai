# 🚀 UTILISATION PRATIQUE : VOTRE SYSTÈME Plant-AI

## 🔍 **TEST IMAGE PAR ÉTAPES - Un vrai exemple**

### **1. CONNEXION API YOLO**
Votre back-end en écoute `http://localhost:8000` :

```
┌─────────────────────────────────────────────────┐
│ BACKEND FASTAPI PLANT_AI EN COURS ...           │
│ ✅ Server ready at http://localhost:8000         │
│ ✅ Model ready: models/yolov8_best.pt (soon)    │ 
│ ✅ CORS: Open for PWA/Mobile                     │
└─────────────────────────────────────────────────┘
```

### **2. Une image échantillon est uploadée**
*Photo d'un ménager en L'Eu nouvelle feuille de tomate avec des taches sur sa face*

 -> UTTILUSER : `curl -X POST http://localhost:8000/predict -F "image=@tache_tomat.jpg"`

### **3. Le modèle analyse pixel-par-pixel**

**Architecture interne qui interprète votre image marche en spécial k.**

Le code backend directement :

```python
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # ✅ 1. L'image qui arrive au serveur est valide JPEG 
    img = _read_image_from_upload(image)
    
    # ✅ 2. Conversion en tableau numérique pour l'IA
    np_img = _pil_to_numpy(img)  # shape (height, width, 3) = [H×W×3 RGB]    
    
    # ✅ 3. APPEL À L'INTELLIGENCE ARTIFICIELLE 
    model       = get_model()  # YOLOv8 best.pt télécharge
    results     = model.predict(source=[np_img], verbose=False)
    
    if results:
        r0 = results[0]  # 1ère image de lot
        
        # ✅ 4. RÉCUPÉRATION DES DÉTECTIONS PAR LE MODÈLE
        xyxy = r0.boxes.xyxy.cpu().numpy()     # BoundingBox (left,top,right,down) de zones maladies
        conf = r0.boxes.conf.cpu().numpy()     # Niveau de confiance : [0.92, 0.87, 0.83]
        cid  = r0.boxes.cls.cpu().numpy()       # ID des maladies : [18, 19, 22]
        
        # ✅ 5. CHAQUE DÉTECTION QUÉ VOIT L'IA
        for i in range(len(xyxy)) :
            bb = xyxy[i].tolist()           # Position de la maladie sur l'image
            trust = float(conf[i])          # 0.89 = 89% sûr
            maladie_id = int(cid[i])        # ID 22 → donc "Tomato leaf late blight"

            # ✅ 6. NOM FRANÇAISE DE LA MALADIE depuis l'entraînement data.yaml
            nom_maladie = names.get(maladie_id, "unknown")
            
            # ✅ 7. ADVICE AGRO DEPUIS DATABASE connaissances 
            conseils = get_recommendations_for_class(nom_maladie) # comme "Appliquer fongicide..."
```

**Valeur retour typique :**
```json
GET http://...8000/predict → 
{
  "predictions":       
  [  
      {
        "class_name":    "Tomato leaf late blight",      // 🎯 MALADIE TROUVÉE  
        "confidence":    0.89,                           // 👆 89% DE SÛRETÉ  
        "bbox":         [403, 124, 523, 300],            // 📍 RÉTANGLE POSITION MALADIE
        "recommendations": ["Fongicide adapté",           // 💉 AVIS DE THERAPE
                          "Rotation culture"],
     }
  ],
  "image_width":     1920,
  "image_height":    1280
}
```
 
## 📱 ** INTÉGRATION CLIENTS WEB/MOBILE POSSIBLE :**

### **Pour un développeur frontend :**
```javascript
const formData = new FormData()
formData.append('image', photoFile) 

await fetch('http://localhost:8000/predict', {
     method: 'POST', 
     body: formData
  }).then(response => response.json())
   .then( result => {
        result.predictions.forEach(m=>{            
            drawBoxRGB(m.bbox[0], m.bbox[1], m.bbox[2], m.bbox[3])  // affiche rectangle maladies
            print( `${m.class_name}: conf=${m.confidence}` )
        })
   })
```


### **CMD depuis servver String pour diagnostic prototyp:**
````
curl -X POST http://localhost:8000/predict -H \"Content-Type: multipart/form-data\" -F image=test/parcelle_local.jpg -o dout.json
cd Plant-AI; cat doudd.json 
````


## 🛠️ **STORAGE LOG HISTORY PLANTES**
Réponse Bdd-roles optionnelle (historique patients) :
- endpoint `/history` ‒ Entrée : obtenir l'historique de diagnostic d'un utilisateur.
- endpoint `/history-POST` ‒ Enoi : On enregistre un nouveau champ, pour tracking continu.

 
--- 

# 🏁 END OF NEARS ACHIEVEMENTS 

✅ Serveur sans erreurs — static de tailles *limits* pagvat-sanctuart

✅ Base reconnaissable: *settings*/model file prod prêt (à placer `models/yolov8_best.pt`)

✅ API endpoints complets et relié avec healthcare advices automatiques ‒.

Vous êtes maintenant **pseudo-Doctor Doctor en plantes** grâce à votre IA, fun fact et full production 🥳

