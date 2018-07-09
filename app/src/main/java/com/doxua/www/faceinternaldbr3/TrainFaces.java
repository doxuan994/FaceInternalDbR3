package com.doxua.www.faceinternaldbr3;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.IntBuffer;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.MatVector;
import static org.bytedeco.javacpp.opencv_core.Point;
import static org.bytedeco.javacpp.opencv_core.Rect;
import static org.bytedeco.javacpp.opencv_core.RectVector;
import static org.bytedeco.javacpp.opencv_core.Scalar;
import static org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import static org.opencv.core.Core.LINE_8;

public class TrainFaces extends AppCompatActivity {
    public static final String TAG = "TrainFaces";
    public static final String EIGEN_FACES_CLASSIFIER = "eigenFacesClassifier.yml";
    public static final String FILE_NAME_PATTERN = "person.%d.%d.jpg";
    public static final int IMG_SIZE = 160;
    public static final int PICK_IMAGE = 100;
    public static final int PHOTOS_TRAIN_QTY = 10;

    // Views.
    private ImageView imageView;
    private TextView textView;

    // Face Detection.
    private CascadeClassifier faceDetector;
    private int absoluteFaceSize = 0;

    // No duplicate photo id.
    // No duplicate person id.
    private int photoIdNumber = 0;
    private int personIdNumber = 0;

    // External storage.
    public static final String EXTERNAL_TRAIN_FOLDER = "saved_images";

    // Internal storage.
    public static final String INTERNAL_TRAIN_FOLDER = "train_folder";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_trainfaces);

        // Create an image view and a text view.
        imageView = (ImageView) findViewById(R.id.imageView);
        textView = (TextView) findViewById(R.id.faces_value);

        // Pick an image for formatting.
        Button pickImageButton = (Button) findViewById(R.id.btnGallery);
        pickImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openGallery();
            }
        });

        // Train button.
        Button trainBtn = (Button) findViewById(R.id.btnTrain);
        trainBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    trainExternalStorageMultipleModels();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });

        // Train multiple button.
        Button trainMultipleBtn = (Button) findViewById(R.id.btnTrainMultiple);
        trainMultipleBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    trainInternalStorageMultipleModels();
                } catch (Exception e) {
                    e.printStackTrace();
                }
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

            // -------------------------------------------------------------------------------------
            //                         DETECT FACES AND DISPLAY THE RESULTS
            // -------------------------------------------------------------------------------------
            detectAndDisplay(bitmap, textView);

            // -------------------------------------------------------------------------------------
            //                                 STORE GRAYSCALE
            // -------------------------------------------------------------------------------------
            try {
                storeImageGrayscaleExternalStorage(bitmap, personIdNumber, photoIdNumber);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Detect faces.
     * Draw a green rectangle around the first detected face.
     * Store color detected faces.
     * Display number of detected faces.
     * Notice: Introducing JavaCV frame converters.
     * http://bytedeco.org/news/2015/04/04/javacv-frame-converters/
     * @param bitmap
     * @param facesValue
     */
    void detectAndDisplay(Bitmap bitmap, TextView facesValue) {
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
        faceDetector = loadClassifierCascade(this, R.raw.frontalface);
        // Detect the face.
        faceDetector.detectMultiScale(greyMat, faces, 1.25f, 3, 1,
                new Size(absoluteFaceSize, absoluteFaceSize),
                new Size(4 * absoluteFaceSize, 4 * absoluteFaceSize));


        // Count number of faces and display in text view.
        int numFaces = (int) faces.size();
        facesValue.setText(Integer.toString(numFaces));

        // -----------------------------------------------------------------------------------------
        //                                         DISPLAY
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
                //                    Convert back to bitmap for displaying
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
    }


    /***********************************************************************************************
     *
     *
     *                                  INTERNAL STORAGE
     *
     *
     **********************************************************************************************/
    /**
     * Save the grayscale format.
     * Theoretically, this method saves images internally but right now let doing a test with external storage.
     * @param bitmap The image to be stored.
     * @param personId
     */
    void storeImageGrayscaleInternalStorage(Bitmap bitmap, int personId, int photoId) throws Exception {
        // Create a new folder internal for our train images.
        File internalFolder = new File(getFilesDir(), INTERNAL_TRAIN_FOLDER);
        if (internalFolder.exists() && !internalFolder.isDirectory()) {
            internalFolder.delete();
        }
        if (!internalFolder.exists()) {
            internalFolder.mkdirs();
        }

        // Create a new gray Mat.
        Mat greyMat = new Mat();
        // JavaCV frame converters.
        AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();

        // -----------------------------------------------------------------------------------------
        //                             CONVERT TO MAT FOR PROCESSING
        // -----------------------------------------------------------------------------------------
        // Convert to Bitmap.
        Frame frame = converterToBitmap.convert(bitmap);
        // Convert to Mat.
        Mat colorMat = converterToMat.convert(frame);

        // Convert to Gray scale.
        cvtColor(colorMat, greyMat, CV_BGR2GRAY);
        // Vector of rectangles where each rectangle contains the detected object.
        RectVector faces = new RectVector();

        // Load the CascadeClassifier class to detect objects.
        faceDetector = loadClassifierCascade(this, R.raw.frontalface);
        // Detect the face.
        faceDetector.detectMultiScale(greyMat, faces, 1.1, 1, 0, new Size(150,150), new Size(500,500));

        // Count number of faces and display in text view.
        int numFaces = (int) faces.size();

        // Save all the detected faces.
        for (int i = 0; i < numFaces; i++) {
            Rect rectFace = faces.get(i);
            rectangle(colorMat, rectFace, new Scalar(0,0,255,0));
            Mat capturedFace = new Mat(greyMat, rectFace);
            resize(capturedFace, capturedFace, new Size(IMG_SIZE,IMG_SIZE));

            // Save an image only if a limitation of the image does not reach.
            if (photoIdNumber < PHOTOS_TRAIN_QTY) {
                File f = new File(internalFolder, String.format(FILE_NAME_PATTERN, personId, photoId));
                f.createNewFile();
                imwrite(f.getAbsolutePath(), capturedFace);
                photoIdNumber++;
            }

            if (photoIdNumber == PHOTOS_TRAIN_QTY) {
                photoIdNumber = 0;
                personIdNumber = 1;
            }
        }
    }

    /**
     * Train our one model with multiple different people faces internal but right now still external for testing purpose.
     * @return
     * @throws Exception
     */
    boolean trainInternalStorageMultipleModels() throws Exception {
        // Find the correct root path where our trained face model will be stored.
        File internalPhotosFolder = new File(getFilesDir(), INTERNAL_TRAIN_FOLDER);
        FilenameFilter imageFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png");
            }
        };

        // Create a list of photo paths.
        File[] files = internalPhotosFolder.listFiles(imageFilter);

        MatVector photos = new MatVector(files.length);
        Mat labels = new Mat(files.length, 1, CV_32SC1);
        IntBuffer rotulosBuffer = labels.createBuffer();

        int counter = 0;
        for (File image: files) {
            Mat photo = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int classe = Integer.parseInt(image.getName().split("\\.")[1]);
            resize(photo, photo, new Size(IMG_SIZE, IMG_SIZE));
            photos.put(counter, photo);
            rotulosBuffer.put(counter, classe);
            counter++;
        }

        // Save our model as YAML file to the folder created on the top of the method.
        if (files.length > 0) {
            FaceRecognizer eigenfaces = EigenFaceRecognizer.create();
            eigenfaces.train(photos, labels);
            File f = new File(internalPhotosFolder, EIGEN_FACES_CLASSIFIER);
            f.createNewFile();
            eigenfaces.save(f.getAbsolutePath());
        }
        return true;
    }

    /***********************************************************************************************
     *
     *
     *                                       EXTERNAL STORAGE
     *
     *
     **********************************************************************************************/
    /**
     * Save the grayscale format.
     * @param bitmap The image to be stored.
     * @param personId
     */
    void storeImageGrayscaleExternalStorage(Bitmap bitmap, int personId, int photoId) throws Exception {
        // Create a new folder external for our train images.
        File myTrainDir = new File(Environment.getExternalStorageDirectory(), EXTERNAL_TRAIN_FOLDER);
        if (myTrainDir.exists() && !myTrainDir.isDirectory()) {
            myTrainDir.delete();
        }
        if (!myTrainDir.exists()) {
            myTrainDir.mkdirs();
        }

        // Create a new gray Mat.
        Mat greyMat = new Mat();
        // JavaCV frame converters.
        AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();

        // -----------------------------------------------------------------------------------------
        //                              CONVERT TO MAT FOR PROCESSING
        // -----------------------------------------------------------------------------------------
        // Convert to Bitmap.
        Frame frame = converterToBitmap.convert(bitmap);
        // Convert to Mat.
        Mat colorMat = converterToMat.convert(frame);

        // Convert to Gray scale.
        cvtColor(colorMat, greyMat, CV_BGR2GRAY);
        // Vector of rectangles where each rectangle contains the detected object.
        RectVector faces = new RectVector();

        // Load the CascadeClassifier class to detect objects.
        faceDetector = loadClassifierCascade(this, R.raw.frontalface);
        // Detect the face.
        faceDetector.detectMultiScale(greyMat, faces, 1.1, 1, 0, new Size(150,150), new Size(500,500));

        // Count number of faces and display in text view.
        int numFaces = (int) faces.size();

        // Save all the detected faces.
        for (int i = 0; i < numFaces; i++) {
            Rect rectFace = faces.get(i);
            rectangle(colorMat, rectFace, new Scalar(0,0,255,0));
            Mat capturedFace = new Mat(greyMat, rectFace);
            resize(capturedFace, capturedFace, new Size(IMG_SIZE,IMG_SIZE));

            // Save an image of limit of images not reach.
            if (photoIdNumber < PHOTOS_TRAIN_QTY) {
                File f = new File(myTrainDir, String.format(FILE_NAME_PATTERN, personId, photoId));
                f.createNewFile();
                imwrite(f.getAbsolutePath(), capturedFace);
                photoIdNumber++;
            }

            if (photoIdNumber == PHOTOS_TRAIN_QTY) {
                photoIdNumber = 0;
                personIdNumber = 1;
            }
        }
    }

    /**
     * Train our one model with multiple different people faces.
     * @return
     * @throws Exception
     */
    boolean trainExternalStorageMultipleModels() throws Exception {
        // Find the correct root path where our trained face model will be stored.
        File photosFolder = new File(Environment.getExternalStorageDirectory(), EXTERNAL_TRAIN_FOLDER);
        FilenameFilter imageFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png");
            }
        };

        // Create a list of photo paths.
        File[] files = photosFolder.listFiles(imageFilter);

        MatVector photos = new MatVector(files.length);
        Mat labels = new Mat(files.length, 1, CV_32SC1);
        IntBuffer rotulosBuffer = labels.createBuffer();

        int counter = 0;
        for (File image: files) {
            Mat photo = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int classe = Integer.parseInt(image.getName().split("\\.")[1]);
            resize(photo, photo, new Size(IMG_SIZE, IMG_SIZE));
            photos.put(counter, photo);
            rotulosBuffer.put(counter, classe);
            counter++;
        }

        // Save our model as YAML file to the folder created on the top of the method.
        if (files.length > 0) {
            FaceRecognizer eigenfaces = EigenFaceRecognizer.create();
            eigenfaces.train(photos, labels);
            File f = new File(photosFolder, EIGEN_FACES_CLASSIFIER);
            f.createNewFile();
            eigenfaces.save(f.getAbsolutePath());
        }
        return true;
    }

    /***********************************************************************************************
     *
     *
     *                                      HELPER METHODS
     *
     *
     **********************************************************************************************/
    /**
     * Load the CascadeClassifier for Face Detection.
     * @param context
     * @param resId
     * @return
     */
    public static CascadeClassifier loadClassifierCascade(Context context, int resId) {
        FileOutputStream fos = null;
        InputStream inputStream;

        inputStream = context.getResources().openRawResource(resId);
        File xmlDir = context.getDir("xml", Context.MODE_PRIVATE);
        File cascadeFile = new File(xmlDir, "temp.xml");
        try {
            fos = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                fos.write(buffer, 0, bytesRead);
            }
        } catch (IOException e) {
            Log.d(TAG, "Can\'t load the cascade file");
            e.printStackTrace();
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        CascadeClassifier detector = new CascadeClassifier(cascadeFile.getAbsolutePath());
        if (detector.isNull()) {
            Log.e(TAG, "Failed to load cascade classifier");
            detector = null;
        } else {
            Log.i(TAG, "Loaded cascade classifier from " + cascadeFile.getAbsolutePath());
        }
        // Delete the temporary directory.
        cascadeFile.delete();
        return detector;
    }
}
