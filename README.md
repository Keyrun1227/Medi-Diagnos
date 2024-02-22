# 🌍🩺Medi-Diagnos
![Image Alt Text](https://github.com/Keyrun1227/Hotel-Booking-Insights-Dashboard/blob/main/Visualization%20of%20Hotel%20Bookings.png)
👩‍⚕️ **Medi-Diagnos**, our Django-powered brainchild designed to revolutionize healthcare. It's not just another project; it's a gateway to precise medical predictions, image-based diagnostics, and a cutting-edge multi-language medical chatbot. Imagine quick, seamless responses in any language, making healthcare accessible at your fingertips! 🌍🩺💡🗣💬

## 🌐 Multilingual Chatbot

Medi-Assistant is a multilingual chatbot that supports communication in various languages. Users can input text in their preferred language, and the chatbot can provide responses in the desired target language.

The frontend is designed to facilitate seamless communication, allowing users to enter text, receive translations, and interact with the chatbot easily. The chatbot utilizes language translation APIs to support multilingual conversations.

## 🏥 Medical Diagnosis

### Objectives

- Predict dental health based on user input.
- Detect pneumonia in chest X-ray images.
- Segment brain MRI images for medical analysis.
- Predict stability for protein sequences.
- Assess the risk of diabetes based on user information.

### Models Used

- Dental Health Prediction: XGBoost Regressor
- Pneumonia Detection: Deep Learning Model for Chest X-ray analysis
- Brain MRI Segmentation: DeepLab Neural Network
- Protein Stability Prediction: BERT (Bidirectional Encoder Representations from Transformers)
- Diabetes Risk Assessment: XGBoost Classifier

## 🖥️ Technologies Used

- **Programming Languages:** Python, JavaScript
- **Frameworks:** Django, TensorFlow, Keras, Scikit-learn
- **Libraries:** NumPy, Pandas, OpenCV, PIL, Transformers, XGBoost
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Docker, Heroku

## 🤖 Chatbot Operation

The chatbot operation involves language translation, medical predictions, and general assistance. Users can communicate with the chatbot by entering text, receiving translations, and obtaining responses related to medical queries.

### Features

- Language Translation: Users can translate text from one language to another.
- Medical Diagnosis: Predict dental health, detect pneumonia, segment brain MRI images, predict protein stability, and assess diabetes risk.

## 🚀 How to Use

1. Enter text in the "Enter text" field.
2. Select the source and target languages.
3. Click the "Get Response From Medi-Assistant" button.
4. View the translated text and responses in the "Translation" field.
5. For specific medical predictions, use the respective features on the website.

## 📁 Project Structure

The project structure includes Django views for medical predictions, language translation functionalities, and a frontend designed for user interaction.

- predict_stability: Predict protein stability using BERT and XGBoost.
- dental_view: Predict dental health using XGBoost.
- predict_pneumonia: Detect pneumonia in chest X-ray images.
- brain_mri_segmentation: Segment brain MRI images using DeepLab.
- predict_diabetes: Assess the risk of diabetes using an XGBoost classifier.
- chatbot.html: Frontend for language translation and chatbot interaction.

## 📚 Dependencies

The project relies on several Python libraries and frameworks, including TensorFlow, Django, Keras, XGBoost, Transformers, and more. Ensure that the necessary dependencies are installed before running the project.

```bash
pip install -r requirements.txt

```

##🌟 Acknowledgments
Special thanks to the developers and contributors of the libraries and frameworks used in this project. The success of Medi-Assistant is attributed to the vibrant open-source community.

Feel free to contribute, provide feedback, or report issues to enhance the capabilities of Medi-Assistant.

👨‍💻 Happy Coding!
