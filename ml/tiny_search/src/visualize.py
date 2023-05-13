#
#
#


def drawdendrograw(clust,labels,jpeg='clusters.jpg'):
     #height and width
     h=getheight(clust)*20
     w=14000
     depth=getdepth(clust)
     scaling=float(w-150)/depth
     img=Image.new('RGB',(w,h),(255,255,255))
     draw=ImageDraw.Draw(img)
     draw.line((0,h/2,10,h/2),fill=(255,0,0))
     drawnode(draw,clust,10,(h/2),scaling,labels)
     img.save(jpeg,'JPEG')
 
def drawnode(draw,clust,x,y,scaling,labels):
    if clust.id<0:
        h1=getheight(clust.left)*20
        h2=getheight(clust.right)*20
        top=y-(h1+h2)/2
        bottom=y+(h1+h2)/2
        #line length
        ll=clust.distance*scaling
        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))
        draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))
        draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))
        drawnode(draw,clust.left,x+ll,top+h1/2,scaling,labels)
        drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,labels)
    else:
        #if this is an endpoint,draw the item label
        draw.text((x+5,y-7),labels[clust.id],(0,0,0))
