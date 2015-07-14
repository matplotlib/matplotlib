# Tools for working with travis-ci
export WHEELHOST="travis-wheels.scikit-image.org"
export WHEELHOUSE="http://${WHEELHOST}/"

retry () {
    # https://gist.github.com/fungusakafungus/1026804
    local retry_max=5
    local count=$retry_max
    while [ $count -gt 0 ]; do
        "$@" && break
        count=$(($count - 1))
        sleep 1
    done

    [ $count -eq 0 ] && {
        echo "Retry failed [$retry_max]: $@" >&2
        return 1
    }
    return 0
}


wheelhouse_pip_install() {
    # Install pip requirements via travis wheelhouse
    retry pip install --timeout=60 --no-index --trusted-host $WHEELHOST --find-links $WHEELHOUSE $@
}
