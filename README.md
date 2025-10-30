# Email Screenshot Extractor

This project is a browser-based tool that uses Optical Character Recognition (OCR) to extract structured data from email screenshots. It is designed to be a self-improving system, allowing users to train a custom machine learning model based on their own data to improve extraction accuracy over time.

## Features

- **In-Browser OCR:** All image processing and text recognition happens directly in your browser. No data is sent to a third-party server for OCR.
- **Rule-Based Extraction:** An initial set of rules and heuristics provides a strong baseline for data extraction.
- **Self-Learning via Feedback:** Users can correct the extracted data, and this corrected data is used to train a custom model.
- **Free and Private:** The entire system is designed to be run for free using browser technologies, Google Apps Script, and Google Colab. Your data remains private.

## How It Works

1.  **Upload:** You upload a screenshot of an email.
2.  **Pre-process:** The image is automatically resized and filtered to improve OCR accuracy.
3.  **OCR:** Tesseract.js reads the text from the image.
4.  **Extract:** The system uses a combination of a rule-based engine and an optional custom-trained model to extract the desired fields.
5.  **Correct & Submit:** You review the extracted data, make any necessary corrections, and submit it. This saves the corrected data for future training and sends it to a configured Google Apps Script endpoint.

## Getting Started

1.  Clone or download this repository.
2.  (Optional) If you have a Google Apps Script Web App, open `config.json` and replace the placeholder URL with your Web App URL.
3.  Open the `index.html` file in your web browser.

## The Self-Learning Loop

This application's key feature is its ability to learn from your corrections. Here's how to use it:

1.  **Process Images:** Use the tool to process your email screenshots. For each one, carefully review and correct the extracted data before clicking **Confirm & Submit**.
2.  **Export Training Data:** After you have processed a good number of images (50-100 is a good start), click the **Export Training Data** button to download a `training_data.json` file.
3.  **Train Your Model:** Follow the instructions in the `TRAINING.md` file to train your own custom model using the exported data. This process is free and uses Google Colab.
4.  **Load Your Model:** Once you've trained and downloaded your `model.json` file, place it in the same directory as `index.html`. The application will automatically load it on the next page refresh.

With a custom model, the extractor will become much more accurate for the specific formats of your emails, combining the power of the general rules with a model trained on your specific data.
