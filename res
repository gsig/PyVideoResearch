{
   cat ../results/$1/checkpoints/model_???.txt 2>/dev/null || 
   cat ../results/$1/model_???.txt 2>/dev/null
} | grep mAP | less
{
   cat ../results/$1/checkpoints/model_???.txt 2>/dev/null || 
   cat ../results/$1/model_???.txt 2>/dev/null 
} | grep mAP | sort | tail -n 1
