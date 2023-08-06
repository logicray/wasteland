package person.me.galaxy.lambda.entity;

import lombok.Getter;
import lombok.Setter;


@Getter
@Setter
public class Artist {
    private String name; //艺术家的名字(例如“甲壳虫乐队”)。
    private String members; //乐队成员(例如“约翰·列侬”)，该字段可为空。 􏰅
    private String origin;  //:乐队来自哪里(例如“利物浦”)。


}
