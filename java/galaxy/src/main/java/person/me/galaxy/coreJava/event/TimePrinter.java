package person.me.galaxy.coreJava.event;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Date;

public class TimePrinter implements ActionListener {
    @Override
    public void actionPerformed(ActionEvent e) {
        System.out.println("at the tone, the time is " + new Date());
        Toolkit.getDefaultToolkit().beep();
    }

    public static void main(String[] args) {
        ActionListener listener = new TimePrinter();
        Timer t = new Timer(5000, listener);
        t.start();
        JOptionPane.showMessageDialog(null, "Quit program?");
        System.exit(0);
    }


}
