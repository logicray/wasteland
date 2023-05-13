package person.me.galaxy.coreJava;

import java.time.DayOfWeek;
import java.time.LocalDate;

public class Section2 {
    public void CalendarPrint() {
        LocalDate localDate = LocalDate.now();
        int thisMonth = localDate.getMonthValue();
        int today = localDate.getDayOfMonth();

        localDate = localDate.minusDays(today - 1);

        DayOfWeek dayOfWeek = localDate.getDayOfWeek();
        int value = dayOfWeek.getValue();
        System.out.println("Mon Tus Wen Thr Fri Sat Sun");
        for (int i = 0; i < value; i++) {
            System.out.print("   ");
        }

        while (localDate.getMonthValue() == thisMonth) {
            System.out.printf("%3d", localDate.getDayOfMonth());
            if (localDate.getDayOfMonth() == today){
                System.out.print("*");
            }else {
                System.out.print(" ");
            }
            localDate = localDate.plusDays(1);
            if (localDate.getDayOfWeek().getValue() == 1){
                System.out.println();
            }
        }
        if (localDate.getDayOfWeek().getValue() != 1){
            System.out.println();
        }
    }

    public static void main(String[] args) {
        Section2 section = new Section2();
        section.CalendarPrint();

    }
}
