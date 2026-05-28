package org.suas.aidetect;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.*;

public class Main {
    public static void main(String[] args) {
        /* Read command line arguments */
        String filePath = null;
        String textFilePath = null;
        for (String arg: args) {
            /* Check if input PDF was provided */
            if (arg.startsWith("--pdf=")) {
                filePath = arg.substring("--pdf=".length());
            }
            /* Check if output TXT was provided */
            if (arg.startsWith("--txt=")) {
                textFilePath = arg.substring("--txt=".length());
            }
        }

        /* Are all arguments provided? */
        if ((filePath == null) || (textFilePath == null)) {
            System.err.println("Usage: java -jar <file>.jar --pdf=<file>.pdf --txt=<file>.txt");
            System.exit(1);
        }

        try {
            // Read in PDF
            InputStream inputStream;
            inputStream = new FileInputStream(filePath);
            File pdfFile = File.createTempFile("pdfextract", ".pdf");
            OutputStream outputStream = new FileOutputStream(pdfFile);
            if (inputStream != null) {
                inputStream.transferTo(outputStream);
                inputStream.close();
            }

            PDDocument document = Loader.loadPDF(pdfFile);

            // Get plain text from PDF
            String text = new PDFTextStripper().getText(document);

            // Store plain text
            FileOutputStream fileOutputStream = new FileOutputStream(textFilePath);
            byte[] bytes = text.getBytes();
            fileOutputStream.write(bytes);
            fileOutputStream.close();

            System.out.println("Done");
        } catch (IOException | NullPointerException e) {
            throw new RuntimeException(e);
        }
    }
}