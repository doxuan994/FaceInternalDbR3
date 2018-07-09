package com.doxua.www.faceinternaldbr3;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.io.File;
import java.io.IOException;

import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.Point;
import static org.bytedeco.javacpp.opencv_core.Rect;
import static org.bytedeco.javacpp.opencv_core.RectVector;
import static org.bytedeco.javacpp.opencv_core.Scalar;
import static org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import static org.opencv.core.Core.LINE_8;


public class RegFaces extends AppCompatActivity {
    public static final String TAG = "RegFaces";
    public static final String INTERNAL_TRAIN_FOLDER = "train_folder";
    private static final int ACCEPT_LEVEL = 1000;
    private static final int MIDDLE_ACCEPT_LEVEL = 2000;
    private static final int PICK_IMAGE = 100;
    private static final int IMG_SIZE = 160;

    // Views.
    private ImageView imageView;
    private TextView tv;
    private TextView result_information;

    // Face Detection.
    private CascadeClassifier faceDetector;
    private int absoluteFaceSize = 0;

    // Face Recognition.
    private FaceRecognizer faceRecognizer = EigenFaceRecognizer.create();

    // External storage.
    public static final String EXTERNAL_TRAIN_FOLDER = "saved_images";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_regfaces);

        // Create the image view and text view.
        imageView = (ImageView) findViewById(R.id.imageView);
        tv = (TextView) findViewById(R.id.predict_faces);
        result_information = findViewById(R.id.result);

        // Pick an image and recognize.
        Button pickImageButton = (Button) findViewById(R.id.btnGallery);
        pickImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openGallery();
            }
        });
    }

    private void openGallery() {
        Intent gallery =
                new Intent(Intent.ACTION_PICK,
                        MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        startActivityForResult(gallery, PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == PICK_IMAGE) {
            Uri imageUri = data.getData();

            // Convert to Bitmap.
            Bitmap bitmap = null;
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
            } catch (IOException e) {
                e.printStackTrace();
            }
            detectDisplayAndRecognize(bitmap);
        }
    }

    /**
     * Face Detection.
     * Face Recognition.
     * Display the detection result and recognition result.
     * @param bitmap
     */
    void detectDisplayAndRecognize(Bitmap bitmap) {
        // Create a new gray Mat.
        Mat greyMat = new Mat();
        // JavaCV frame converters.
        AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();

        // -----------------------------------------------------------------------------------------
        //                              Convert to mat for processing
        // -----------------------------------------------------------------------------------------
        // Convert to Bitmap.
        Frame frame = converterToBitmap.convert(bitmap);
        // Convert to Mat.
        Mat colorMat = converterToMat.convert(frame);

        // Convert to Gray scale.
        cvtColor(colorMat, greyMat, CV_BGR2GRAY);
        // Vector of rectangles where each rectangle contains the detected object.
        RectVector faces = new RectVector();

        // -----------------------------------------------------------------------------------------
        //                                  FACE DETECTION
        // -----------------------------------------------------------------------------------------
        // Load the CascadeClassifier class to detect objects.
        faceDetector = TrainFaces.loadClassifierCascade(RegFaces.this, R.raw.frontalface);
        // Detect the face.
        faceDetector.detectMultiScale(greyMat, faces, 1.25f, 3, 1,
                new Size(absoluteFaceSize, absoluteFaceSize),
                new Size(4 * absoluteFaceSize, 4 * absoluteFaceSize));

        // Count number of faces.
        int numFaces = (int) faces.size();

        // -----------------------------------------------------------------------------------------
        //                                       DISPLAY
        // -----------------------------------------------------------------------------------------
        if ( numFaces > 0 ) {
            // Multiple face detection.
            for (int i = 0; i < numFaces; i++) {

                int x = faces.get(i).x();
                int y = faces.get(i).y();
                int w = faces.get(i).width();
                int h = faces.get(i).height();

                rectangle(colorMat, new Point(x, y), new Point(x + w, y + h), Scalar.GREEN, 2, LINE_8, 0);

                // ---------------------------------------------------------------------------------
                //                      Convert back to bitmap for displaying
                // ---------------------------------------------------------------------------------
                // Convert processed Mat back to a Frame
                frame = converterToMat.convert(colorMat);
                // Copy the data to a Bitmap for display or something
                Bitmap bm = converterToBitmap.convert(frame);

                // Display the picked image.
                imageView.setImageBitmap(bm);
            }
        } else {
            imageView.setImageBitmap(bitmap);
        }

        // -----------------------------------------------------------------------------------------
        //                                  FACE RECOGNITION
        // -----------------------------------------------------------------------------------------
        recognizeMultipleInternalStorage(faces.get(0), greyMat, tv);
    }

    /***********************************************************************************************
     *
     *
     *                                  INTERNAL STORAGE
     *
     *
     **********************************************************************************************/
    /**
     * Predict using one model only but can predict faces of different people.
     * Recognize multiple faces using only one model.
     * prediction = 0 Angelina Jolie
     * prediction = 1 Tom Cruise
     * IMPORTANT!
     * @param dadosFace
     * @param greyMat
     */
    void recognizeMultipleInternalStorage(Rect dadosFace, Mat greyMat, TextView tv) {
        int personId = 0;
        String personName = "";

        // Find the correct root path where our trained face model is stored.
        File internalPhotosFolder = new File(getFilesDir(), INTERNAL_TRAIN_FOLDER);
        File f = new File(internalPhotosFolder, TrainFaces.EIGEN_FACES_CLASSIFIER);

        // Loads a persisted model and state from a given XML or YAML file.
        faceRecognizer.read(f.getAbsolutePath());

        Mat detectedFace = new Mat(greyMat, dadosFace);
        resize(detectedFace, detectedFace, new Size(IMG_SIZE, IMG_SIZE));

        IntPointer label = new IntPointer(1);
        DoublePointer reliability = new DoublePointer(1);
        faceRecognizer.predict(detectedFace, label, reliability);

        // Display on the text view what we found.
        int prediction = label.get(0);
        int acceptanceLevel = (int) reliability.get(0);

        if (prediction == 0) {
            personName = "Angelina Jolie";
            personId = 1;
        }

        if (prediction == 1) {
            personName = "Tom Cruise";
            personId = 2;
        }

        // -----------------------------------------------------------------------------------------
        //                         DISPLAY THE FACE RECOGNITION PREDICTION
        // -----------------------------------------------------------------------------------------
        if ((prediction != 0 && prediction != 1) || acceptanceLevel > MIDDLE_ACCEPT_LEVEL)
        {
            // Display on text view, not matching or unknown person.
            tv.setText("Unknown");
            result_information.setText("");
        }
        else if (acceptanceLevel >= ACCEPT_LEVEL && acceptanceLevel <= MIDDLE_ACCEPT_LEVEL)
        {
            tv.setText(
                    "Found a match but not sure." +
                            "\nWarning! Acceptable Level is high!" +
                            "\nHi, " + personName +  " " + acceptanceLevel +
                            "\nPerson ID: " + personId +
                            "\nPrediction Id: " + prediction
            );
            result_information.setText("");
        }
        else
        {
            // Display the information for the matching image.
            tv.setText(
                    "A match is found." +
                            "\nHi, " + personName +  " " + acceptanceLevel +
                            "\nPerson ID: " + personId +
                            "\nPrediction Id: " + prediction
            );

            if (personId >= 1) {
                DatabaseAccess databaseAccess = DatabaseAccess.getInstance(getApplicationContext());
                databaseAccess.open();

                String info = databaseAccess.getInformation(personId);
                result_information.setText(info);

                databaseAccess.close();
            }
        }
    }


    /***********************************************************************************************
     *
     *
     *                                       EXTERNAL STORAGE
     *
     *
     **********************************************************************************************/


    /**
     * Predict using one model only but can predict faces of different people.
     * Recognize multiple faces using only one model.
     * prediction = 0 Angelina Jolie
     * prediction = 1 Tom Cruise
     * IMPORTANT!
     * @param dadosFace
     * @param greyMat
     */
    void recognizeMultipleExternalStorage(Rect dadosFace, Mat greyMat, TextView tv) {
        int personId = 0;
        String personName = "";

        // Find the correct root path where our trained face model is stored.
        File photosFolder = new File(Environment.getExternalStorageDirectory(), INTERNAL_TRAIN_FOLDER);
        File f = new File(photosFolder, TrainFaces.EIGEN_FACES_CLASSIFIER);

        // Loads a persisted model and state from a given XML or YAML file.
        faceRecognizer.read(f.getAbsolutePath());

        Mat detectedFace = new Mat(greyMat, dadosFace);
        resize(detectedFace, detectedFace, new Size(IMG_SIZE, IMG_SIZE));

        IntPointer label = new IntPointer(1);
        DoublePointer reliability = new DoublePointer(1);
        faceRecognizer.predict(detectedFace, label, reliability);

        // Display on the text view what we found.
        int prediction = label.get(0);
        int acceptanceLevel = (int) reliability.get(0);

        if (prediction == 0) {
            personName = "Angelina Jolie";
            personId = 1;
        }

        if (prediction == 1) {
            personName = "Tom Cruise";
            personId = 2;
        }

        // -----------------------------------------------------------------------------------------
        //                         DISPLAY THE FACE RECOGNITION PREDICTION
        // -----------------------------------------------------------------------------------------
        if ((prediction != 0 && prediction != 1) || acceptanceLevel > MIDDLE_ACCEPT_LEVEL)
        {
            // Display on text view, not matching or unknown person.
            tv.setText("Unknown");
            result_information.setText("");
        }
        else if (acceptanceLevel >= ACCEPT_LEVEL && acceptanceLevel <= MIDDLE_ACCEPT_LEVEL)
        {
            tv.setText(
                    "Found a match but not sure." +
                            "\nWarning! Acceptable Level is high!" +
                            "\nHi, " + personName +  " " + acceptanceLevel +
                            "\nPerson ID: " + personId +
                            "\nPrediction Id: " + prediction
            );
            result_information.setText("");
        }
        else
        {
            // Display the information for the matching image.
            tv.setText(
                    "A match is found." +
                            "\nHi, " + personName +  " " + acceptanceLevel +
                            "\nPerson ID: " + personId +
                            "\nPrediction Id: " + prediction
            );

            if (personId >= 1) {
                DatabaseAccess databaseAccess = DatabaseAccess.getInstance(getApplicationContext());
                databaseAccess.open();

                String info = databaseAccess.getInformation(personId);
                result_information.setText(info);

                databaseAccess.close();
            }
        }
    }
}
