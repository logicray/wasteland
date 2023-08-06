package person.me.galaxy.util;


public class TestFileUtil {


    public static void main(String[] args) {
        String fileName = "/temp/newTemp.txt";
        FileUtil.readFileByBytes(fileName);
        FileUtil.readFileByChars(fileName);
        FileUtil.readFileByLines(fileName);
        FileUtil.readFileByRandomAccess(fileName);


        //String fileName = "C:/temp/newTemp.txt";
        String content = "new append!";
        //按方法A追加文件
        FileUtil.appendMethodA(fileName, content);
        FileUtil.appendMethodA(fileName, "append end. \n");
        //显示文件内容
        FileUtil.readFileByLines(fileName);
        //按方法B追加文件
        FileUtil.appendMethodB(fileName, content);
        FileUtil.appendMethodB(fileName, "append end. \n");
        //显示文件内容
        FileUtil.readFileByLines(fileName);
    }
}

