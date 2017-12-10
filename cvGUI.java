package KKH.Opencv;

// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

import javax.swing.*;
import java.io.*;
import javax.imageio.*;
import java.awt.image.BufferedImage;
import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.DataBufferByte;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

public class cvGUI {

    // show an image on a Java Swing frame/window
    public static void imshow(Mat src){
        BufferedImage bufImage = null;
        try {
            // convert Mat image to byte array
            MatOfByte matOfByte = new MatOfByte();
            Imgcodecs.imencode(".jpg", src, matOfByte);
            byte[] byteArray = matOfByte.toArray();
            // form input stream from byte array to form buffered image
            InputStream in = new ByteArrayInputStream(byteArray);
            bufImage = ImageIO.read(in);

            // new frame and display image there
            JFrame frame = new JFrame("Image");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.getContentPane().setLayout(new FlowLayout());
            frame.getContentPane().add(new JLabel(new ImageIcon(bufImage)));
            frame.pack();
            frame.setVisible(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // show an image on a Java Swing frame/window
    // adapted from official opencv sample code
    public static void imshow(String winTitle, Mat src, int x, int y){
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if ( src.channels() > 1 ) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = src.channels()*src.cols()*src.rows();
        byte [] b = new byte[bufferSize];
        src.get(0,0, b); // get all the pixels
        BufferedImage img = new BufferedImage(src.cols(),src.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);

        ImageIcon icon=new ImageIcon(img);
        JFrame frame=new JFrame(winTitle);
        JLabel lbl=new JLabel(icon);
        frame.add(lbl);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setLocation(x, y);
        frame.setVisible(true);
    }

    public static void imshow(String winTitle, Mat src){
        imshow(winTitle, src, 0, 0);
    }

}
