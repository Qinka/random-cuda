{-# LANGUAGE ForeignFunctionInterface #-}

module Binding.HQinkaLC where

import           Control.Monad
import           Data.Vector.Storable as V
import           Data.Word
import           Foreign.C
import           Foreign.ForeignPtr
import           Foreign.Ptr

linearCombination :: Float -- ^ coeffient 1
                  -> Vector Word8 -- ^ vector 1
                  -> Float -- ^ coeffient 2
                  -> Vector Word8 -- ^ vector 2
                  -> IO (Vector Word8,Int)
linearCombination coe1 v1 coe2 v2 = do
    let (fp1,ln1) = unsafeToForeignPtr0 v1
        (fp2,ln2) = unsafeToForeignPtr0 v2
    when (ln1 /= ln2) $ fail "shape dose not same"
    fp3 <- mallocForeignPtrArray ln1
    i <- ga fp1 fp2 fp3 $ \p1 p2 p3 ->
        _linearCombination coe1 p1 coe2 p2 ln1 p3
    return (unsafeFromForeignPtr0 fp3 ln1,  i)
    where ga fp1 fp2 fp3 func =
            withForeignPtr fp1 $ \p1 ->
                withForeignPtr fp2 (withForeignPtr fp3 . func p1)



foreign import ccall "linearCombination" _linearCombination :: Float -> Ptr Word8 -> Float -> Ptr Word8 -> Int -> Ptr Word8 -> IO Int

foreign import ccall "qinkalc_link_test" qinkaLinkTest :: IO ()
