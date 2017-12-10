package KKH.Opencv;

import KKH.StdLib.Matk;
import KKH.StdLib.Matkc;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/*
Various static methods to make calling opencv more convenient & powerful
*/

public final class cvUtilFuncs {

    // convert i,j,k 3D position in a matrix to linear index stored in row major format
    public static int pos_to_RM_lin_index(int nr, int nc, int nch, int i, int j, int k)
    {
        return i*nch*nc + j*nch + k;
    }

    /**
     * Similar to Matlab's im2single.
     * @param img Mat (an image) of depth 8U. Can have any number channels.
     * @return Mat of depth 32F. Same number of channels as input.
     */
    public static Mat im2single(Mat img)
    {
        Mat imgOut = new Mat(img.size(), CvType.makeType(CvType.CV_32F, img.channels()));
        img.convertTo(imgOut, CvType.CV_32F, 1/255.0);
        return imgOut;
    }

    /**
     * Similar to Matlab's im2double.
     * @param img Mat (an image) of depth 8U. Can have any number channels.
     * @return Mat of depth 64F. Same number of channels as input.
     */
    public static Mat im2double(Mat img)
    {
        Mat imgOut = new Mat(img.size(), CvType.makeType(CvType.CV_64F, img.channels()));
        img.convertTo(imgOut, CvType.CV_64F, 1/255.0);
        return imgOut;
    }

    public static int[] Mat8U_to_ArrayInt(Mat m)
    {
        if(m.depth() != CvType.CV_8U)
            throw new IllegalArgumentException("ERROR: Mat is not of type CV_8U");

        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        int[] aL = new int [nr * nc * nch];

        byte[] data = new byte[nr*nc*nch];
        m.get(0, 0, data);

        for(int i=0; i<nr*nc*nch; i++)
            aL[i] = data[i] & 0xFF;

        return aL;
    }

    public static byte[] Mat8U_to_ArrayByte(Mat m)
    {
        if(m.depth() != CvType.CV_8U)
            throw new IllegalArgumentException("ERROR: Mat is not of type CV_8U");

        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        byte[] data = new byte[nr*nc*nch];
        m.get(0, 0, data);

        return data;
    }

    public static float[] Mat32F_to_Array(Mat m)
    {
        if(m.depth() != CvType.CV_32F)
            throw new IllegalArgumentException("ERROR: Mat is not of type CV_32F");

        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        float[] data = new float[nr*nc*nch];
        m.get(0, 0, data);

        return data;
    }

    public static double[] Mat64F_to_Array(Mat m)
    {
        if(m.depth() != CvType.CV_64F)
            throw new IllegalArgumentException("ERROR: Mat is not of type CV_64F");

        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        double[] data = new double[nr*nc*nch];
        m.get(0, 0, data);

        return data;
    }

    public static int[] Mat32S_to_Array(Mat m)
    {
        if(m.depth() != CvType.CV_32S)
            throw new IllegalArgumentException("ERROR: Mat is not of type CV32S");

        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        int[] data = new int[nr*nc*nch];
        m.get(0, 0, data);

        return data;
    }

    public static Mat Array_to_Mat(byte[] arr, int nrows, int ncols, int nchannels)
    {
        Mat m = new Mat(nrows, ncols, CvType.makeType(CvType.CV_8U, nchannels));
        m.put(0, 0, arr);
        return m;
    }

    public static Mat Array_to_Mat(float[] arr, int nrows, int ncols, int nchannels)
    {
        Mat m = new Mat(nrows, ncols, CvType.makeType(CvType.CV_32F, nchannels));
        m.put(0, 0, arr);
        return m;
    }

    public static Mat Array_to_Mat(double[] arr, int nrows, int ncols, int nchannels)
    {
        Mat m = new Mat(nrows, ncols, CvType.makeType(CvType.CV_64F, nchannels));
        m.put(0, 0, arr);
        return m;
    }

    public static Mat Array_to_Mat(int[] arr, int nrows, int ncols, int nchannels)
    {
        Mat m = new Mat(nrows, ncols, CvType.makeType(CvType.CV_32S, nchannels));
        m.put(0, 0, arr);
        return m;
    }

    public static int[][][] Mat8U_to_3DArrayInt(Mat m)
    {
        if(m.depth() != CvType.CV_8U)
            throw new IllegalArgumentException("ERROR: Mat is not of type CV_8U");

        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        int[][][] aL = new int [nch][nr][nc];

        byte[] data = new byte[nr*nc*nch];
        m.get(0, 0, data);

        for(int k=0; k<nch; k++)
        {
            int[][] arr_cur = aL[k];

            for(int i=0; i<nr; i++)
            {
                for(int j=0; j<nc; j++)
                {
                    arr_cur[i][j] = data[pos_to_RM_lin_index(nr, nc, nch, i, j, k)] & 0xFF;
                }
            }
        }

        return aL;
    }

    public static byte[][][] Mat8U_to_3DArrayByte(Mat m)
    {
        if(m.depth() != CvType.CV_8U)
            throw new IllegalArgumentException("ERROR: Mat is not of type CV_8U");

        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        byte[][][] aL = new byte [nch][nr][nc];

        byte[] data = new byte[nr*nc*nch];
        m.get(0, 0, data);

        for(int k=0; k<nch; k++)
        {
            byte[][] arr_cur = aL[k];

            for(int i=0; i<nr; i++)
            {
                for(int j=0; j<nc; j++)
                {
                    arr_cur[i][j] = data[pos_to_RM_lin_index(nr, nc, nch, i, j, k)];
                }
            }
        }

        return aL;
    }

    public static float[][][] Mat32F_to_3DArray(Mat m)
    {
        if(m.depth() != CvType.CV_32F)
            throw new IllegalArgumentException("ERROR: Mat is not of type CV_32F");

        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        float[][][] aL = new float[nch][nr][nc];

        float[] data = new float[nr*nc*nch];
        m.get(0, 0, data);

        for(int k=0; k<nch; k++)
        {
            float[][] arr_cur = aL[k];

            for(int i=0; i<nr; i++)
            {
                for(int j=0; j<nc; j++)
                {
                    arr_cur[i][j] = data[pos_to_RM_lin_index(nr, nc, nch, i, j, k)];
                }
            }
        }

        return aL;
    }

    public static double[][][] Mat64F_to_3DArray(Mat m)
    {
        if(m.depth() != CvType.CV_64F)
            throw new IllegalArgumentException("ERROR: Mat is not of type CV_64F");

        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        double[][][] aL = new double[nch][nr][nc];

        double[] data = new double[nr*nc*nch];
        m.get(0, 0, data);

        for(int k=0; k<nch; k++)
        {
            double[][] arr_cur = aL[k];

            for(int i=0; i<nr; i++)
            {
                for(int j=0; j<nc; j++)
                {
                    arr_cur[i][j] = data[pos_to_RM_lin_index(nr, nc, nch, i, j, k)];
                }
            }
        }

        return aL;
    }

    public static int[][][] Mat32S_to_3DArray(Mat m)
    {
        if(m.depth() != CvType.CV_32S)
            throw new IllegalArgumentException("ERROR: Mat is not of type CV_32S");

        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        int[][][] aL = new int[nch][nr][nc];

        int[] data = new int[nr*nc*nch];
        m.get(0, 0, data);

        for(int k=0; k<nch; k++)
        {
            int[][] arr_cur = aL[k];

            for(int i=0; i<nr; i++)
            {
                for(int j=0; j<nc; j++)
                {
                    arr_cur[i][j] = data[pos_to_RM_lin_index(nr, nc, nch, i, j, k)];
                }
            }
        }

        return aL;
    }

    public static Matk cvMat_to_Matk(Mat m)
    {
        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        Matk mOut;

        switch(m.depth())
        {
            case CvType.CV_32S:
                int[] data_int = new int[nr*nc*nch];
                m.get(0, 0, data_int);
                mOut = new Matk(data_int, false, nr, nc, nch);
                break;
            case CvType.CV_32F:
                float[] data_float = new float[nr*nc*nch];
                m.get(0, 0, data_float);
                mOut = new Matk(data_float, false, nr, nc, nch);
                break;
            case CvType.CV_64F:
                double[] data_double = new double[nr*nc*nch];
                m.get(0, 0, data_double);
                mOut = new Matk(data_double, false, nr, nc, nch);
                break;
            case CvType.CV_8U:
                byte[] data_byte = new byte[nr*nc*nch];
                m.get(0, 0, data_byte);
                mOut = new Matk(data_byte, false, nr, nc, nch);
                break;
            default:
                throw new IllegalArgumentException("ERROR: md != CvType.CV_32S && md != CvType.CV_32F && md != CvType.CV_64F && md != CvType.CV_8U");
        }

        return mOut;
    }

    public static Matkc cvMat_to_Matkc(Mat m)
    {
        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        Matkc mOut;

        switch(m.depth())
        {
            case CvType.CV_32S:
                int[] data_int = new int[nr*nc*nch];
                m.get(0, 0, data_int);
                mOut = new Matkc(data_int, false, nr, nc, nch);
                break;
            case CvType.CV_32F:
                float[] data_float = new float[nr*nc*nch];
                m.get(0, 0, data_float);
                mOut = new Matkc(data_float, false, nr, nc, nch);
                break;
            case CvType.CV_64F:
                double[] data_double = new double[nr*nc*nch];
                m.get(0, 0, data_double);
                mOut = new Matkc(data_double, false, nr, nc, nch);
                break;
            case CvType.CV_8U:
                byte[] data_byte = new byte[nr*nc*nch];
                m.get(0, 0, data_byte);
                mOut = new Matkc(data_byte, false, nr, nc, nch);
                break;
            default:
                throw new IllegalArgumentException("ERROR: md != CvType.CV_32S && md != CvType.CV_32F && md != CvType.CV_64F && md != CvType.CV_8U");
        }

        return mOut;
    }

    public static Mat Matk_to_cvMat(Matk m, String as_type)
    {
        int nr = m.nrows();
        int nc = m.ncols();
        int nch = m.nchannels();

        Mat mOut;
        int cc;

        switch(as_type)
        {
            case "int":
                mOut = new Mat(nr, nc, CvType.makeType(CvType.CV_32S, nch));
                int[] data_int = new int[nr * nc * nch];
                cc = 0;
                for(int i=0; i<nr; i++)
                    for(int j=0; j<nc; j++)
                        for(int k=0; k<nch; k++)
                            data_int[cc++] = (int)m.get(i,j,k);
                mOut.put(0, 0, data_int);
                break;
            case "double":
                mOut = new Mat(nr, nc, CvType.makeType(CvType.CV_64F, nch));
                double[] data_double = new double[nr * nc * nch];
                cc = 0;
                for(int i=0; i<nr; i++)
                    for(int j=0; j<nc; j++)
                        for(int k=0; k<nch; k++)
                            data_double[cc++] = m.get(i,j,k);
                mOut.put(0, 0, data_double);
                break;
            case "float":
                mOut = new Mat(nr, nc, CvType.makeType(CvType.CV_32F, nch));
                float[] data_float = new float[nr * nc * nch];
                cc = 0;
                for(int i=0; i<nr; i++)
                    for(int j=0; j<nc; j++)
                        for(int k=0; k<nch; k++)
                            data_float[cc++] = (float)m.get(i,j,k);
                mOut.put(0, 0, data_float);
                break;
            case "byte":
                mOut = new Mat(nr, nc, CvType.makeType(CvType.CV_8U, nch));
                byte[] data_byte = new byte[nr * nc * nch];
                cc = 0;
                for(int i=0; i<nr; i++)
                    for(int j=0; j<nc; j++)
                        for(int k=0; k<nch; k++)
                            data_byte[cc++] = (byte)m.get(i,j,k);
                mOut.put(0, 0, data_byte);
                break;
            default:
                throw new IllegalArgumentException("ERROR: as_type mst be \"int\", \"double\", \"float\" or \"byte\"");
        }

        return mOut;
    }

    public static Mat Matkc_to_cvMat(Matkc m, String as_type)
    {
        int nr = m.nrows();
        int nc = m.ncols();
        int nch = m.nchannels();

        Mat mOut;
        int cc;

        switch(as_type)
        {
            case "int":
                mOut = new Mat(nr, nc, CvType.makeType(CvType.CV_32S, nch));
                int[] data_int = new int[nr * nc * nch];
                cc = 0;
                for(int i=0; i<nr; i++)
                    for(int j=0; j<nc; j++)
                        for(int k=0; k<nch; k++)
                            data_int[cc++] = (int)m.get(i,j,k);
                mOut.put(0, 0, data_int);
                break;
            case "double":
                mOut = new Mat(nr, nc, CvType.makeType(CvType.CV_64F, nch));
                double[] data_double = new double[nr * nc * nch];
                cc = 0;
                for(int i=0; i<nr; i++)
                    for(int j=0; j<nc; j++)
                        for(int k=0; k<nch; k++)
                            data_double[cc++] = m.get(i,j,k);
                mOut.put(0, 0, data_double);
                break;
            case "float":
                mOut = new Mat(nr, nc, CvType.makeType(CvType.CV_32F, nch));
                float[] data_float = new float[nr * nc * nch];
                cc = 0;
                for(int i=0; i<nr; i++)
                    for(int j=0; j<nc; j++)
                        for(int k=0; k<nch; k++)
                            data_float[cc++] = (float)m.get(i,j,k);
                mOut.put(0, 0, data_float);
                break;
            case "byte":
                mOut = new Mat(nr, nc, CvType.makeType(CvType.CV_8U, nch));
                byte[] data_byte = new byte[nr * nc * nch];
                cc = 0;
                for(int i=0; i<nr; i++)
                    for(int j=0; j<nc; j++)
                        for(int k=0; k<nch; k++)
                            data_byte[cc++] = (byte)m.get(i,j,k);
                mOut.put(0, 0, data_byte);
                break;
            default:
                throw new IllegalArgumentException("ERROR: as_type mst be \"int\", \"double\", \"float\" or \"byte\"");
        }

        return mOut;
    }




    public static void print_mat(Mat m)
    {
        int nr = m.rows();
        int nc = m.cols();
        int nch = m.channels();

        System.out.println("=========== Printing opencv matrix ============");

        for(int k=0; k<nch; k++)
        {
            System.out.println("Channel num " + (k+1));
            for(int i=0; i<nr; i++)
            {
                for(int j=0; j<nc; j++)
                {
                    switch(m.depth())
                    {
                        case CvType.CV_8U: {
                            byte[] data = new byte[nch];
                            m.get(i, j, data);
                            System.out.print(data[k] + "\t");
                            break;
                        }
                        case CvType.CV_64F:
                        {
                            double[] data = new double[nch];
                            m.get(i, j, data);
                            System.out.print(data[k] + "\t");
                            break;
                        }
                        case CvType.CV_32F:
                        {
                            float[] data = new float[nch];
                            m.get(i, j, data);
                            System.out.print(data[k] + "\t");
                            break;
                        }
                        case CvType.CV_32S:
                        {
                            int[] data = new int[nch];
                            m.get(i, j, data);
                            System.out.print(data[k] + "\t");
                            break;
                        }
                        case CvType.CV_16S:
                        {
                            short[] data = new short[nch];
                            m.get(i, j, data);
                            System.out.print(data[k] + "\t");
                            break;
                        }

                        default:
                            throw new IllegalArgumentException("Unknown opencv mat type for printing.");

                    }
                }

                System.out.println();
            }
        }

        System.out.println("=========== Opencv matrix printed ============");
    }
}
