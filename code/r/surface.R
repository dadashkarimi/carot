library(png) 
library(RCurl) 
library("ggimage")
library(grid)
library(ggpubr)
library(gridExtra)
library(patchwork)

dict <- new.env(hash = TRUE, parent = emptyenv(), size = NA)
dict[["shen"]] <- "Shen"
dict[["shen368"]] <- "Shen368"
dict[["schaefer"]] <- "Shae"
dict[["craddock400"]] <- "Crad400"
dict[["craddock"]] <- "Crad"
dict[["brainnetome"]] <- "Btm246"
dict[["power"]] <- "Power"
dict[["dosenbach"]] <- "Dsnbch"
dict[["relational"]] <- "rel"
dict[["emotion"]] <- "emtn"
dict[["wm"]] <- "wm"
dict[["motor"]] <- "motor"
dict[["gambling"]] <- "gam"
dict[["lang"]] <- "lang"
dict[["social"]] <- "soc"
dict[["rest1"]] <- "rest1"
atlas_list <- c("schaefer","craddock","brainnetome","power","dosenbach","craddock400")

my_plot<-function(source,target,task,id){
  f0 = paste("surface_c0_",source,"_",target,"_",task,".png",sep="")
  f1 = paste("surface_c1_",source,"_",target,"_",task,".png",sep="")
  # f2 = paste("surface_c2_",source,"_",target,"_",task,".png",sep="")
  img1 <- readPNG(f0)
  img2 <- readPNG(f1)
  # img3 <- readPNG(f2)
  
  g1 <- rasterGrob(img1, interpolate=TRUE)
  g2 <- rasterGrob(img2, interpolate=TRUE)
  # g3 <- rasterGrob(img3, interpolate=TRUE)
  g1<-qplot()+annotation_custom(g1)+theme_void(base_size = 1)+labs(title="",subtitle= dict[[source]],caption = "")+ theme(plot.subtitle  = element_text(hjust = 0.5,size=8))
  g2<-qplot()+annotation_custom(g2)+theme_void(base_size = 1)+labs(title="",caption = "",subtitle= dict[[target]])+ theme(plot.subtitle  = element_text(hjust = 0.5,size=8))
  # g3<-qplot()+annotation_custom(g3)+theme_void(base_size = 7)+labs(title="",caption="",subtitle= bquote(T["#"~ .(dict[[source]])] ~ "=" ~.(dict[[target]])))+ theme(plot.subtitle  = element_text(hjust = 0.5,size=15))
  if(id==1){
    g1<-g1#+labs(title="Preprocessing (left and middle) and OT (right)")
  }
  g<-ggarrange(g1,g2,ncol=2,widths = c(1,1))
  return(g)
}
source<-"schaefer"
print(source)
target<-"shen"
task<-"rest1"

Cairo::Cairo(18,10,
             file = paste(paste("surface_all_",target,task,sep="_"),".png",sep=""),
             type = "png",bg = "white", dpi = 300, units = "cm" #you can change to pixels etc
)

my_plots <- list()
p_index <-1
# atlas_list <- c("schaefer","craddock")

for(source in atlas_list){
  if(source!=target){
    g<-my_plot(source,target,task,p_index)
    my_plots[[p_index]]<-g
    p_index <- p_index+1
  }
}

plot.new()
ggarrange(plotlist = my_plots,nrow = 5,ncol = 2,heights = c(1,1,1,1))
#  theme(plot.margin = unit(c(3,3,3,3), "lines"))


# 
# do.call("grid.arrange", c(my_plots, nrow=3))
# 
legend("bottomright",
       pch=21, pt.cex=1.0,
       pt.bg=c('red','blue'),
       legend=c('source', 'target'),
       cex=0.8, bty='n')

dev.off()
# }