# copy this to root directory of data and ./normalize-resample.sh
# https://github.com/mindslab-ai/voicefilter/blob/master/utils/normalize-resample.sh

open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}
run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    printf '%.3d' $? >&3
    )&
}

N=32 # set "N" as your CPU core number.
open_sem $N
for f in $(find . -name "*.flac"); do
    # convert to 16 khz mono
    run_with_lock ffmpeg -y -i "$f" -ar 16000 -ac 1 "${f%.*}.wav"
done
