package person.me.galaxy;


import org.apache.commons.cli.*;


/**
 * main function,
 */
public class App {

    public static void main(String[] args) {
        //参数解析
        CommandLineParser commandLineParser = new DefaultParser();
        Options OPTIONS = new Options();
        // help
        OPTIONS.addOption(Option.builder("h").longOpt("help").type(String.class).desc("usage help").build());
        // config
        OPTIONS.addOption(Option.builder("c").hasArg(true).longOpt("config").type(String.class).desc("location of the config file").build());
        //当config加载或者解析报错时，直接打印报错信息，并退出
        try {
            CommandLine commandLine = commandLineParser.parse(OPTIONS, args);
            String configPath = commandLine.getOptionValue("config","");
            System.out.println(configPath);
        } catch (ParseException e) {
            System.out.println("parse error:" + e.getMessage());
            System.exit(-1);
        }
    }


}
