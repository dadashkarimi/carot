library(fmsb)

# Create data: note in High school for several students
set.seed(99)

df_268 <- read.table("268.txt", header = TRUE,sep = "\t")
df_368 <- read.table("368.txt", header = TRUE,sep = "\t")

colnames(df_268) <-c("100","200","300","400","500","600")
colnames(df_368) <-c("100","200","300","400","500","600")


data <- as.data.frame(matrix( sample( 0:20 , 15 , replace=F) , ncol=5))
colnames(data) <- c("math" , "english" , "biology" , "music" , "R-coding" )
rownames(data) <- paste("mister" , letters[1:3] , sep="-")

# To use the fmsb package, I have to add 2 lines to the dataframe: the max and min of each variable to show on the plot!
data <- rbind(rep(20,5) , rep(0,5) , data)


df_268 <- rbind(rep(0.45,6) , rep(0.55,6) , df_268)
df_368 <- rbind(rep(0.48,6) , rep(0.55,6) , df_368)

# plot with default options:


colors_in=c( "red", "black" , "blue","yellow" )

pdf(width = 9,file = "368.pdf")

radarchart(df_368,pcol=colors_in)

legend(x=1.2, y=1, legend = c( "Low", "Medium" , "High","Full" ), bty = "n", pch=20 , col=colors_in , text.col = "grey", cex=1.2, pt.cex=3)

dev.off()

