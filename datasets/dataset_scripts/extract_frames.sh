#!/bin/bash
 
SEEDFOLDER=indir/
tmpfile=$(mktemp tmp/tmplist.XXXXXX)
find $SEEDFOLDER -maxdepth 1 -type f | shuf > $tmpfile
MAXW=320
MAXH=320
while read -u 10 line; do 
    filename=$(basename $line)
    output="outdir/${filename%.*}_1.jpg"
    outputdir="outdir/${filename%.*}"
    if [ ! -d $outputdir ]; then
	    mkdir $outputdir
	    ffmpeg -i "$line" -qscale:v 3 -filter:v "scale='if(gt(a,$MAXW/$MAXH),$MAXW,-1)':'if(gt(a,$MAXW/$MAXH),-1,$MAXH)',fps=fps=24" "outdir/${filename%.*}/${filename%.*}_%0d.jpg";
    fi
done 10<$tmpfile
rm $tmpfile
