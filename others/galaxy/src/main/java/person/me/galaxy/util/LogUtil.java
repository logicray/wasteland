package person.me.galaxy.util;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

public class LogUtil {

    public static String getExceptionStack(Exception ex) {
        ByteArrayOutputStream out;
        PrintStream pout = null;
        String ret;
        try {
            out = new ByteArrayOutputStream();
            pout = new PrintStream(out);
            ex.printStackTrace(pout);
            ret = new String(out.toByteArray());
            out.close();
        } catch (Exception e) {
            return ex.getMessage();
        } finally {
            if (pout != null) {
                pout.close();
            }
        }
        return ret;
    }

}
