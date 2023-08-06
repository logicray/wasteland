package person.me.galaxy.coreJava.network;

import java.io.IOException;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Arrays;
import java.util.Scanner;

/**
 * @version 0.0.1 2021-06-21
 */
public class N1 {
    public static void main(String[] args) throws IOException {
//        socket();
        inet();
    }

    private static void socket() throws IOException {
        try (Socket s = new Socket("time-a.nist.gov", 13)){
            Scanner in = new Scanner(s.getInputStream(), "UTF-8");

            while (in.hasNext()){
                String line = in.nextLine();
                System.out.println(line);
            }
        }
    }

    private static void inet() throws UnknownHostException {
        InetAddress address = InetAddress.getByName("baidu.com");
        System.out.println(Arrays.toString(address.getAddress()));
        System.out.println(address);
    }
}
