#!/bin/bash

SCRIPT_REPO="https://code.videolan.org/videolan/libbluray.git"
SCRIPT_COMMIT="bb5bc108ec695889855f06df338958004ff289ef"

ffbuild_enabled() {
    return 0
}

ffbuild_dockerbuild() {
    git-mini-clone "$SCRIPT_REPO" "$SCRIPT_COMMIT" libbluray
    cd libbluray

    if [[ $TARGET == mac* ]]; then
        gsed -i 's/dec_init/libbluray_dec_init/g' src/libbluray/disc/*.c src/libbluray/disc/*.h
    else
        sed -i 's/dec_init/libbluray_dec_init/g' src/libbluray/disc/*.c src/libbluray/disc/*.h
    fi

    ./bootstrap

    local myconf=(
        --prefix="$FFBUILD_PREFIX"
        --disable-shared
        --enable-static
        --with-pic
        --disable-doxygen-doc
        --disable-doxygen-dot
        --disable-doxygen-html
        --disable-doxygen-ps
        --disable-doxygen-pdf
        --disable-examples
        --disable-bdjava-jar
    )

    if [[ $TARGET == win* || $TARGET == linux* ]]; then
        myconf+=(
            --host="$FFBUILD_TOOLCHAIN"
        )
    elif [[ $TARGET == mac* ]]; then
        myconf+=(
            --disable-dependency-tracking
        )
    else
        echo "Unknown target"
        return -1
    fi

    ./configure "${myconf[@]}"
    make -j$(nproc)
    make install
}

ffbuild_configure() {
    echo --enable-libbluray
}

ffbuild_unconfigure() {
    echo --disable-libbluray
}
