library(fmsb)
library(Rcpp)
library(ggstatsplot)
library(palmerpenguins)
library(tidyverse)
library(gridExtra)
library(ggpubr)
library(RColorBrewer)
# library(ggradar)

# Create data: note in High school for several students
set.seed(99)
transpose_matrix <- function(df) {
  df = df[1:6,]
  a <-t(df)
  colnames(a) <-c("100","200","300","400","500","600")
  a<-a[1:4,]
  return(a)
}


atlas_list <- c("shen","shen368","schaefer","craddock","craddock400","brainnetome","power","dosenbach")
# atlas_list <- c("shen","schaefer","craddock","brainnetome","power","dosenbach")

target ="shen"
atlas_list <- atlas_list[atlas_list!=target]

source1 =atlas_list[1]
source2 =atlas_list[2]
source3 =atlas_list[3]
source4 =atlas_list[4]
source5 =atlas_list[5]
source6 =atlas_list[6]
source7 =atlas_list[7]



s1 <- read.table(paste("data/",paste(source1,target,sep="_"),"_frame_size_corr.txt",sep=""), header = TRUE,sep = " ")
s2 <- read.table(paste("data/",paste(source2,target,sep="_"),"_frame_size_corr.txt",sep=""), header = TRUE,sep = " ")
s3 <- read.table(paste("data/",paste(source3,target,sep="_"),"_frame_size_corr.txt",sep=""), header = TRUE,sep = " ")
s4 <- read.table(paste("data/",paste(source4,target,sep="_"),"_frame_size_corr.txt",sep=""), header = TRUE,sep = " ")
s5 <- read.table(paste("data/",paste(source5,target,sep="_"),"_frame_size_corr.txt",sep=""), header = TRUE,sep = " ")
s6 <- read.table(paste("data/",paste(source6,target,sep="_"),"_frame_size_corr.txt",sep=""), header = TRUE,sep = " ")
s7 <- read.table(paste("data/",paste(source7,target,sep="_"),"_frame_size_corr.txt",sep=""), header = TRUE,sep = " ")

s1<-transpose_matrix(s1)
s2<-transpose_matrix(s2)
s3<-transpose_matrix(s3)
s4<-transpose_matrix(s4)
s5<-transpose_matrix(s5)
s6<-transpose_matrix(s6)
s7<-transpose_matrix(s7)

# s1<-s1[1:4,]
# s2<-s2[1:4,]

df_s1 <-  rbind(rep(max(s1),6) , rep(min(s1),6),s1)
df_s2 <-  rbind(rep(max(s2),6) , rep(min(s2),6), s2)
df_s3 <-  rbind(rep(max(s3),6) , rep(min(s3),6),s3)
df_s4 <-  rbind(rep(max(s4),6) , rep(min(s4),6),s4)
df_s5 <-  rbind(rep(max(s5),6) , rep(min(s5),6),s5)
df_s6 <-  rbind(rep(max(s6),6) , rep(min(s6),6),s6)
df_s7 <-  rbind(rep(max(s7),6) , rep(min(s7),6),s7)

rownames(df_s1)<-c(1:6)
rownames(df_s2)<-c(1:6)
rownames(df_s3)<-c(1:6)
rownames(df_s4)<-c(1:6)
rownames(df_s5)<-c(1:6)
rownames(df_s6)<-c(1:6)
rownames(df_s7)<-c(1:6)

colors_in=c( "red", "black" , "blue","yellow","purple" )

paletter_vector <- paletteer::paletteer_d(palette = "palettetown::venusaur",
                                          n = nlevels(as.factor(atlas_list)),type = "discrete")
                                          
# pdf(width = 4,file = "268.pdf")
par(mar=c(1, 2, 2, 1)) #decrease default margin
layout(matrix(1:4, ncol=2)) #draw 4 plots to device

# p1<-radarchart(as.data.frame.matrix(df_s1),pcol=paletter_vector[1:6])
# p2<-radarchart(as.data.frame.matrix(df_s2),pcol=paletter_vector[1:6])
# p3<-radarchart(as.data.frame.matrix(df_s3),pcol=paletter_vector[1:6])
# p4<-radarchart(as.data.frame.matrix(df_s4),pcol=paletter_vector[1:6])
# p5<-radarchart(as.data.frame.matrix(df_s5),pcol=paletter_vector[1:6])
# p6<-radarchart(as.data.frame.matrix(df_s6),pcol=paletter_vector[1:6])
# p7<-radarchart(as.data.frame.matrix(df_s7),pcol=paletter_vector[1:6])

Cairo::Cairo(40,10,
             file = paste("figs/",paste("all",target,"pentagon",sep="_"), ".png", sep = ""),
             type = "png",bg = "transparent", dpi = 300, units = "cm" #you can change to pixels etc
)


# p5<-radarchart(as.data.frame.matrix(df_s5),pcol=paletter_vector[1:5])
#par(mar=c(0, 5, 1, 0)) #decrease default margin
par(mfrow=c(1,7), mai = c(1, 0.1, 0.1, 0.1)) 

#layout(matrix(1:6, ncol=3)) #draw 4 plots to device
paletter_vector = brewer.pal(n = 4, name = 'RdBu')
radarchart(as.data.frame.matrix(df_s1),pcol=paletter_vector[1:4], vlcex=1.8,plwd=2,cglwd=1.1,cglcol="black",cex.main=2,title=target)
mtext(side = 1, line = 2.5, at = 0, cex = 1.5, source1, font = 2)


radarchart(as.data.frame.matrix(df_s2),pcol=paletter_vector[1:4],vlcex=1.8,plwd=2,cglwd=1.1,cglcol="black")
mtext(side = 1, line = 2.5, at = 0, cex = 1.5, source2, font = 1)


radarchart(as.data.frame.matrix(df_s3),pcol=paletter_vector[1:4],vlcex=1.8,plwd=2,cglwd=1.1,cglcol="black")
mtext(side = 1, line = 2.5, at = 0, cex = 1.5, source3, font = 1)


radarchart(as.data.frame.matrix(df_s4),pcol=paletter_vector[1:4],vlcex=1.8,plwd=2,cglwd=1.1,cglcol="black")
mtext(side = 1, line = 2.5, at = 0, cex = 1.5, source4, font = 1)


radarchart(as.data.frame.matrix(df_s5),pcol=paletter_vector[1:4],vlcex=1.8,plwd=2,cglwd=1.1,cglcol="black")
mtext(side = 1, line = 2.5, at = 0, cex = 1.5, source5, font = 1)



radarchart(as.data.frame.matrix(df_s6),pcol=paletter_vector[1:4],vlcex=1.8,plwd=2,cglwd=1.1,cglcol="black")
mtext(side = 1, line = 2.5, at = 0, cex = 1.5, source6, font = 1)

radarchart(as.data.frame.matrix(df_s7),pcol=paletter_vector[1:4],vlcex=1.8,plwd=2,cglwd=1.1,cglcol="black")
mtext(side = 1, line = 2.5, at = 0, cex = 1.5, source7, font = 1)

legend(x=0.5, y=1.9, legend = c( "Low", "Medium" , "High","Full" ), 
       bty = "n", pch=20 , col=paletter_vector[1:4] , text.col = "black", cex=1.0, pt.cex=3)

# radarchart(as.data.frame.matrix(df_s5),pcol=paletter_vector[1:6])




dev.off()

