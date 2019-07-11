d=20190101
while [ "$d" != 20190131 ]
    do python LIMRAD94_to_Cloudnet_v2.py date=${d} path=/lacroshome/remsens_lim/data/cloudnet/punta-arenas/calibrated/limrad94/2019/
    d=$(date -d "$d + 1 day" '+%Y%m%d')
done
