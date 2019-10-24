for((i=1;i<=6;i++))
do
  python train_all_area.py   --test_area  ${i}  --log_dir log/Area${i}  > train_and_test_area${i}.out >&1	
done
