       БK"	  @_ЕеоAbrain.Event:2`║м.?▄     О(ц
	КW_ЕеоA"▓И
ї
inputPlaceholder*
dtype0*9
_output_shapes'
%:#                  Я*.
shape%:#                  Я
ѕ
PlaceholderPlaceholder*)
shape :                  X*
dtype0*4
_output_shapes"
 :                  X
і
Placeholder_1Placeholder*)
shape :                  X*
dtype0*4
_output_shapes"
 :                  X
N
Placeholder_2Placeholder*
shape: *
dtype0
*
_output_shapes
: 
Е
.conv1/weights/Initializer/random_uniform/shapeConst*%
valueB"             * 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
:
Њ
,conv1/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *─oЙ* 
_class
loc:@conv1/weights
Њ
,conv1/weights/Initializer/random_uniform/maxConst*
valueB
 *─o>* 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 
­
6conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv1/weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *

seed**
T0* 
_class
loc:@conv1/weights*
seed2
м
,conv1/weights/Initializer/random_uniform/subSub,conv1/weights/Initializer/random_uniform/max,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*
_output_shapes
: 
В
,conv1/weights/Initializer/random_uniform/mulMul6conv1/weights/Initializer/random_uniform/RandomUniform,conv1/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
я
(conv1/weights/Initializer/random_uniformAdd,conv1/weights/Initializer/random_uniform/mul,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
│
conv1/weights
VariableV2*
dtype0*&
_output_shapes
: *
shared_name * 
_class
loc:@conv1/weights*
	container *
shape: 
М
conv1/weights/AssignAssignconv1/weights(conv1/weights/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
ђ
conv1/weights/readIdentityconv1/weights*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
Ў
.conv1/biases/Initializer/zeros/shape_as_tensorConst*
valueB: *
_class
loc:@conv1/biases*
dtype0*
_output_shapes
:
і
$conv1/biases/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@conv1/biases
н
conv1/biases/Initializer/zerosFill.conv1/biases/Initializer/zeros/shape_as_tensor$conv1/biases/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@conv1/biases*
_output_shapes
: 
Ў
conv1/biases
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@conv1/biases*
	container *
shape: 
║
conv1/biases/AssignAssignconv1/biasesconv1/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
q
conv1/biases/readIdentityconv1/biases*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
i
Kelz/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
У
Kelz/conv1/Conv2DConv2Dinputconv1/weights/read*
paddingSAME*9
_output_shapes'
%:#                  Я *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ъ
Kelz/conv1/BiasAddBiasAddKelz/conv1/Conv2Dconv1/biases/read*
data_formatNHWC*9
_output_shapes'
%:#                  Я *
T0
o
Kelz/conv1/ReluReluKelz/conv1/BiasAdd*
T0*9
_output_shapes'
%:#                  Я 
Е
.conv2/weights/Initializer/random_uniform/shapeConst*%
valueB"              * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
:
Њ
,conv2/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *.щСй* 
_class
loc:@conv2/weights
Њ
,conv2/weights/Initializer/random_uniform/maxConst*
valueB
 *.щС=* 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
: 
­
6conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv2/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv2/weights*
seed2*
dtype0*&
_output_shapes
:  *

seed*
м
,conv2/weights/Initializer/random_uniform/subSub,conv2/weights/Initializer/random_uniform/max,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*
_output_shapes
: 
В
,conv2/weights/Initializer/random_uniform/mulMul6conv2/weights/Initializer/random_uniform/RandomUniform,conv2/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
:  
я
(conv2/weights/Initializer/random_uniformAdd,conv2/weights/Initializer/random_uniform/mul,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
:  
│
conv2/weights
VariableV2*
shared_name * 
_class
loc:@conv2/weights*
	container *
shape:  *
dtype0*&
_output_shapes
:  
М
conv2/weights/AssignAssignconv2/weights(conv2/weights/Initializer/random_uniform*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0* 
_class
loc:@conv2/weights
ђ
conv2/weights/readIdentityconv2/weights*&
_output_shapes
:  *
T0* 
_class
loc:@conv2/weights
i
Kelz/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ы
Kelz/conv2/Conv2DConv2DKelz/conv1/Reluconv2/weights/read*9
_output_shapes'
%:#                  Я *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
g
Kelz/conv2/BatchNorm/ConstConst*
valueB *  ђ?*
dtype0*
_output_shapes
: 
Е
6conv2/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB: *'
_class
loc:@conv2/BatchNorm/beta
џ
,conv2/BatchNorm/beta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *'
_class
loc:@conv2/BatchNorm/beta
З
&conv2/BatchNorm/beta/Initializer/zerosFill6conv2/BatchNorm/beta/Initializer/zeros/shape_as_tensor,conv2/BatchNorm/beta/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@conv2/BatchNorm/beta*
_output_shapes
: 
Е
conv2/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@conv2/BatchNorm/beta*
	container *
shape: 
┌
conv2/BatchNorm/beta/AssignAssignconv2/BatchNorm/beta&conv2/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@conv2/BatchNorm/beta*
validate_shape(*
_output_shapes
: 
Ѕ
conv2/BatchNorm/beta/readIdentityconv2/BatchNorm/beta*
T0*'
_class
loc:@conv2/BatchNorm/beta*
_output_shapes
: 
и
=conv2/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB: *.
_class$
" loc:@conv2/BatchNorm/moving_mean*
dtype0*
_output_shapes
:
е
3conv2/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@conv2/BatchNorm/moving_mean*
dtype0*
_output_shapes
: 
љ
-conv2/BatchNorm/moving_mean/Initializer/zerosFill=conv2/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor3conv2/BatchNorm/moving_mean/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@conv2/BatchNorm/moving_mean*
_output_shapes
: 
и
conv2/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@conv2/BatchNorm/moving_mean*
	container *
shape: 
Ш
"conv2/BatchNorm/moving_mean/AssignAssignconv2/BatchNorm/moving_mean-conv2/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@conv2/BatchNorm/moving_mean
ъ
 conv2/BatchNorm/moving_mean/readIdentityconv2/BatchNorm/moving_mean*
_output_shapes
: *
T0*.
_class$
" loc:@conv2/BatchNorm/moving_mean
Й
@conv2/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB: *2
_class(
&$loc:@conv2/BatchNorm/moving_variance*
dtype0*
_output_shapes
:
»
6conv2/BatchNorm/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ђ?*2
_class(
&$loc:@conv2/BatchNorm/moving_variance*
dtype0*
_output_shapes
: 
Ю
0conv2/BatchNorm/moving_variance/Initializer/onesFill@conv2/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor6conv2/BatchNorm/moving_variance/Initializer/ones/Const*
T0*

index_type0*2
_class(
&$loc:@conv2/BatchNorm/moving_variance*
_output_shapes
: 
┐
conv2/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes
: *
shared_name *2
_class(
&$loc:@conv2/BatchNorm/moving_variance*
	container *
shape: 
Ё
&conv2/BatchNorm/moving_variance/AssignAssignconv2/BatchNorm/moving_variance0conv2/BatchNorm/moving_variance/Initializer/ones*
T0*2
_class(
&$loc:@conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
ф
$conv2/BatchNorm/moving_variance/readIdentityconv2/BatchNorm/moving_variance*
T0*2
_class(
&$loc:@conv2/BatchNorm/moving_variance*
_output_shapes
: 
_
Kelz/conv2/BatchNorm/Const_1Const*
dtype0*
_output_shapes
: *
valueB 
_
Kelz/conv2/BatchNorm/Const_2Const*
valueB *
dtype0*
_output_shapes
: 
М
#Kelz/conv2/BatchNorm/FusedBatchNormFusedBatchNormKelz/conv2/Conv2DKelz/conv2/BatchNorm/Constconv2/BatchNorm/beta/readKelz/conv2/BatchNorm/Const_1Kelz/conv2/BatchNorm/Const_2*
epsilon%oЃ:*
T0*
data_formatNHWC*Q
_output_shapes?
=:#                  Я : : : : *
is_training(
a
Kelz/conv2/BatchNorm/Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *oЃ:
Д
)Kelz/conv2/BatchNorm/AssignMovingAvg/readIdentityconv2/BatchNorm/moving_mean*
T0*.
_class$
" loc:@conv2/BatchNorm/moving_mean*
_output_shapes
: 
о
(Kelz/conv2/BatchNorm/AssignMovingAvg/SubSub)Kelz/conv2/BatchNorm/AssignMovingAvg/read%Kelz/conv2/BatchNorm/FusedBatchNorm:1*
_output_shapes
: *
T0*.
_class$
" loc:@conv2/BatchNorm/moving_mean
╠
(Kelz/conv2/BatchNorm/AssignMovingAvg/MulMul(Kelz/conv2/BatchNorm/AssignMovingAvg/SubKelz/conv2/BatchNorm/Const_3*
T0*.
_class$
" loc:@conv2/BatchNorm/moving_mean*
_output_shapes
: 
Я
$Kelz/conv2/BatchNorm/AssignMovingAvg	AssignSubconv2/BatchNorm/moving_mean(Kelz/conv2/BatchNorm/AssignMovingAvg/Mul*
T0*.
_class$
" loc:@conv2/BatchNorm/moving_mean*
_output_shapes
: *
use_locking( 
▒
+Kelz/conv2/BatchNorm/AssignMovingAvg_1/readIdentityconv2/BatchNorm/moving_variance*
T0*2
_class(
&$loc:@conv2/BatchNorm/moving_variance*
_output_shapes
: 
я
*Kelz/conv2/BatchNorm/AssignMovingAvg_1/SubSub+Kelz/conv2/BatchNorm/AssignMovingAvg_1/read%Kelz/conv2/BatchNorm/FusedBatchNorm:2*
T0*2
_class(
&$loc:@conv2/BatchNorm/moving_variance*
_output_shapes
: 
н
*Kelz/conv2/BatchNorm/AssignMovingAvg_1/MulMul*Kelz/conv2/BatchNorm/AssignMovingAvg_1/SubKelz/conv2/BatchNorm/Const_3*
T0*2
_class(
&$loc:@conv2/BatchNorm/moving_variance*
_output_shapes
: 
В
&Kelz/conv2/BatchNorm/AssignMovingAvg_1	AssignSubconv2/BatchNorm/moving_variance*Kelz/conv2/BatchNorm/AssignMovingAvg_1/Mul*
use_locking( *
T0*2
_class(
&$loc:@conv2/BatchNorm/moving_variance*
_output_shapes
: 
ђ
Kelz/conv2/ReluRelu#Kelz/conv2/BatchNorm/FusedBatchNorm*9
_output_shapes'
%:#                  Я *
T0
┼
Kelz/pool2/MaxPoolMaxPoolKelz/conv2/Relu*
ksize
*
paddingVALID*9
_output_shapes'
%:#                  ░ *
T0*
data_formatNHWC*
strides

d
Kelz/dropout2/dropout/keep_probConst*
valueB
 *  ђ>*
dtype0*
_output_shapes
: 
m
Kelz/dropout2/dropout/ShapeShapeKelz/pool2/MaxPool*
_output_shapes
:*
T0*
out_type0
m
(Kelz/dropout2/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
(Kelz/dropout2/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
╩
2Kelz/dropout2/dropout/random_uniform/RandomUniformRandomUniformKelz/dropout2/dropout/Shape*
T0*
dtype0*9
_output_shapes'
%:#                  ░ *
seed2I*

seed*
ц
(Kelz/dropout2/dropout/random_uniform/subSub(Kelz/dropout2/dropout/random_uniform/max(Kelz/dropout2/dropout/random_uniform/min*
T0*
_output_shapes
: 
Л
(Kelz/dropout2/dropout/random_uniform/mulMul2Kelz/dropout2/dropout/random_uniform/RandomUniform(Kelz/dropout2/dropout/random_uniform/sub*
T0*9
_output_shapes'
%:#                  ░ 
├
$Kelz/dropout2/dropout/random_uniformAdd(Kelz/dropout2/dropout/random_uniform/mul(Kelz/dropout2/dropout/random_uniform/min*9
_output_shapes'
%:#                  ░ *
T0
Ф
Kelz/dropout2/dropout/addAddKelz/dropout2/dropout/keep_prob$Kelz/dropout2/dropout/random_uniform*
T0*9
_output_shapes'
%:#                  ░ 
Ѓ
Kelz/dropout2/dropout/FloorFloorKelz/dropout2/dropout/add*9
_output_shapes'
%:#                  ░ *
T0
Ю
Kelz/dropout2/dropout/divRealDivKelz/pool2/MaxPoolKelz/dropout2/dropout/keep_prob*
T0*9
_output_shapes'
%:#                  ░ 
ю
Kelz/dropout2/dropout/mulMulKelz/dropout2/dropout/divKelz/dropout2/dropout/Floor*9
_output_shapes'
%:#                  ░ *
T0
Е
.conv3/weights/Initializer/random_uniform/shapeConst*%
valueB"          @   * 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
:
Њ
,conv3/weights/Initializer/random_uniform/minConst*
valueB
 *║З║й* 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
: 
Њ
,conv3/weights/Initializer/random_uniform/maxConst*
valueB
 *║З║=* 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
: 
­
6conv3/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv3/weights/Initializer/random_uniform/shape*

seed**
T0* 
_class
loc:@conv3/weights*
seed2T*
dtype0*&
_output_shapes
: @
м
,conv3/weights/Initializer/random_uniform/subSub,conv3/weights/Initializer/random_uniform/max,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*
_output_shapes
: 
В
,conv3/weights/Initializer/random_uniform/mulMul6conv3/weights/Initializer/random_uniform/RandomUniform,conv3/weights/Initializer/random_uniform/sub*&
_output_shapes
: @*
T0* 
_class
loc:@conv3/weights
я
(conv3/weights/Initializer/random_uniformAdd,conv3/weights/Initializer/random_uniform/mul,conv3/weights/Initializer/random_uniform/min*&
_output_shapes
: @*
T0* 
_class
loc:@conv3/weights
│
conv3/weights
VariableV2*
shared_name * 
_class
loc:@conv3/weights*
	container *
shape: @*
dtype0*&
_output_shapes
: @
М
conv3/weights/AssignAssignconv3/weights(conv3/weights/Initializer/random_uniform*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv3/weights
ђ
conv3/weights/readIdentityconv3/weights*
T0* 
_class
loc:@conv3/weights*&
_output_shapes
: @
Ў
.conv3/biases/Initializer/zeros/shape_as_tensorConst*
valueB:@*
_class
loc:@conv3/biases*
dtype0*
_output_shapes
:
і
$conv3/biases/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@conv3/biases*
dtype0*
_output_shapes
: 
н
conv3/biases/Initializer/zerosFill.conv3/biases/Initializer/zeros/shape_as_tensor$conv3/biases/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@conv3/biases*
_output_shapes
:@
Ў
conv3/biases
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv3/biases
║
conv3/biases/AssignAssignconv3/biasesconv3/biases/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes
:@
q
conv3/biases/readIdentityconv3/biases*
T0*
_class
loc:@conv3/biases*
_output_shapes
:@
i
Kelz/conv3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ч
Kelz/conv3/Conv2DConv2DKelz/dropout2/dropout/mulconv3/weights/read*9
_output_shapes'
%:#                  ░@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
ъ
Kelz/conv3/BiasAddBiasAddKelz/conv3/Conv2Dconv3/biases/read*
T0*
data_formatNHWC*9
_output_shapes'
%:#                  ░@
o
Kelz/conv3/ReluReluKelz/conv3/BiasAdd*
T0*9
_output_shapes'
%:#                  ░@
─
Kelz/pool3/MaxPoolMaxPoolKelz/conv3/Relu*
ksize
*
paddingVALID*8
_output_shapes&
$:"                  X@*
T0*
data_formatNHWC*
strides

d
Kelz/dropout3/dropout/keep_probConst*
valueB
 *  ђ>*
dtype0*
_output_shapes
: 
m
Kelz/dropout3/dropout/ShapeShapeKelz/pool3/MaxPool*
T0*
out_type0*
_output_shapes
:
m
(Kelz/dropout3/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
(Kelz/dropout3/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
╔
2Kelz/dropout3/dropout/random_uniform/RandomUniformRandomUniformKelz/dropout3/dropout/Shape*
T0*
dtype0*8
_output_shapes&
$:"                  X@*
seed2j*

seed*
ц
(Kelz/dropout3/dropout/random_uniform/subSub(Kelz/dropout3/dropout/random_uniform/max(Kelz/dropout3/dropout/random_uniform/min*
_output_shapes
: *
T0
л
(Kelz/dropout3/dropout/random_uniform/mulMul2Kelz/dropout3/dropout/random_uniform/RandomUniform(Kelz/dropout3/dropout/random_uniform/sub*
T0*8
_output_shapes&
$:"                  X@
┬
$Kelz/dropout3/dropout/random_uniformAdd(Kelz/dropout3/dropout/random_uniform/mul(Kelz/dropout3/dropout/random_uniform/min*
T0*8
_output_shapes&
$:"                  X@
ф
Kelz/dropout3/dropout/addAddKelz/dropout3/dropout/keep_prob$Kelz/dropout3/dropout/random_uniform*
T0*8
_output_shapes&
$:"                  X@
ѓ
Kelz/dropout3/dropout/FloorFloorKelz/dropout3/dropout/add*
T0*8
_output_shapes&
$:"                  X@
ю
Kelz/dropout3/dropout/divRealDivKelz/pool3/MaxPoolKelz/dropout3/dropout/keep_prob*
T0*8
_output_shapes&
$:"                  X@
Џ
Kelz/dropout3/dropout/mulMulKelz/dropout3/dropout/divKelz/dropout3/dropout/Floor*8
_output_shapes&
$:"                  X@*
T0
c

Kelz/ShapeShapeKelz/dropout3/dropout/mul*
T0*
out_type0*
_output_shapes
:
b
Kelz/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
d
Kelz/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
d
Kelz/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
њ
Kelz/strided_sliceStridedSlice
Kelz/ShapeKelz/strided_slice/stackKelz/strided_slice/stack_1Kelz/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
d
Kelz/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
f
Kelz/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
Kelz/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
џ
Kelz/strided_slice_1StridedSlice
Kelz/ShapeKelz/strided_slice_1/stackKelz/strided_slice_1/stack_1Kelz/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
X
Kelz/flatten4/shape/2Const*
dtype0*
_output_shapes
: *
value
B :ђ,
ќ
Kelz/flatten4/shapePackKelz/strided_sliceKelz/strided_slice_1Kelz/flatten4/shape/2*
T0*

axis *
N*
_output_shapes
:
ќ
Kelz/flatten4ReshapeKelz/dropout3/dropout/mulKelz/flatten4/shape*
T0*
Tshape0*5
_output_shapes#
!:                  ђ,
Ю
,fc5/weights/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@fc5/weights*
dtype0*
_output_shapes
:
Ј
*fc5/weights/Initializer/random_uniform/minConst*
valueB
 *з5й*
_class
loc:@fc5/weights*
dtype0*
_output_shapes
: 
Ј
*fc5/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *з5=*
_class
loc:@fc5/weights
т
4fc5/weights/Initializer/random_uniform/RandomUniformRandomUniform,fc5/weights/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ђ,ђ*

seed**
T0*
_class
loc:@fc5/weights*
seed2Ђ
╩
*fc5/weights/Initializer/random_uniform/subSub*fc5/weights/Initializer/random_uniform/max*fc5/weights/Initializer/random_uniform/min*
T0*
_class
loc:@fc5/weights*
_output_shapes
: 
я
*fc5/weights/Initializer/random_uniform/mulMul4fc5/weights/Initializer/random_uniform/RandomUniform*fc5/weights/Initializer/random_uniform/sub*
T0*
_class
loc:@fc5/weights* 
_output_shapes
:
ђ,ђ
л
&fc5/weights/Initializer/random_uniformAdd*fc5/weights/Initializer/random_uniform/mul*fc5/weights/Initializer/random_uniform/min*
T0*
_class
loc:@fc5/weights* 
_output_shapes
:
ђ,ђ
Б
fc5/weights
VariableV2*
_class
loc:@fc5/weights*
	container *
shape:
ђ,ђ*
dtype0* 
_output_shapes
:
ђ,ђ*
shared_name 
┼
fc5/weights/AssignAssignfc5/weights&fc5/weights/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@fc5/weights*
validate_shape(* 
_output_shapes
:
ђ,ђ
t
fc5/weights/readIdentityfc5/weights*
T0*
_class
loc:@fc5/weights* 
_output_shapes
:
ђ,ђ
ќ
,fc5/biases/Initializer/zeros/shape_as_tensorConst*
valueB:ђ*
_class
loc:@fc5/biases*
dtype0*
_output_shapes
:
є
"fc5/biases/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@fc5/biases*
dtype0*
_output_shapes
: 
═
fc5/biases/Initializer/zerosFill,fc5/biases/Initializer/zeros/shape_as_tensor"fc5/biases/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@fc5/biases*
_output_shapes	
:ђ
Ќ

fc5/biases
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *
_class
loc:@fc5/biases*
	container *
shape:ђ
│
fc5/biases/AssignAssign
fc5/biasesfc5/biases/Initializer/zeros*
T0*
_class
loc:@fc5/biases*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
l
fc5/biases/readIdentity
fc5/biases*
T0*
_class
loc:@fc5/biases*
_output_shapes	
:ђ
e
Kelz/fc5/Tensordot/ShapeShapeKelz/flatten4*
T0*
out_type0*
_output_shapes
:
Y
Kelz/fc5/Tensordot/RankConst*
value	B :*
dtype0*
_output_shapes
: 
a
Kelz/fc5/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
c
!Kelz/fc5/Tensordot/GreaterEqual/yConst*
dtype0*
_output_shapes
: *
value	B : 
љ
Kelz/fc5/Tensordot/GreaterEqualGreaterEqualKelz/fc5/Tensordot/axes!Kelz/fc5/Tensordot/GreaterEqual/y*
T0*
_output_shapes
:
t
Kelz/fc5/Tensordot/CastCastKelz/fc5/Tensordot/GreaterEqual*

SrcT0
*
_output_shapes
:*

DstT0
t
Kelz/fc5/Tensordot/mulMulKelz/fc5/Tensordot/CastKelz/fc5/Tensordot/axes*
T0*
_output_shapes
:
[
Kelz/fc5/Tensordot/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
x
Kelz/fc5/Tensordot/LessLessKelz/fc5/Tensordot/axesKelz/fc5/Tensordot/Less/y*
T0*
_output_shapes
:
n
Kelz/fc5/Tensordot/Cast_1CastKelz/fc5/Tensordot/Less*

SrcT0
*
_output_shapes
:*

DstT0
t
Kelz/fc5/Tensordot/addAddKelz/fc5/Tensordot/axesKelz/fc5/Tensordot/Rank*
T0*
_output_shapes
:
w
Kelz/fc5/Tensordot/mul_1MulKelz/fc5/Tensordot/Cast_1Kelz/fc5/Tensordot/add*
_output_shapes
:*
T0
v
Kelz/fc5/Tensordot/add_1AddKelz/fc5/Tensordot/mulKelz/fc5/Tensordot/mul_1*
T0*
_output_shapes
:
`
Kelz/fc5/Tensordot/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
`
Kelz/fc5/Tensordot/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
б
Kelz/fc5/Tensordot/rangeRangeKelz/fc5/Tensordot/range/startKelz/fc5/Tensordot/RankKelz/fc5/Tensordot/range/delta*
_output_shapes
:*

Tidx0
Д
Kelz/fc5/Tensordot/ListDiffListDiffKelz/fc5/Tensordot/rangeKelz/fc5/Tensordot/add_1*
T0*
out_idx0*2
_output_shapes 
:         :         
Х
Kelz/fc5/Tensordot/GatherGatherKelz/fc5/Tensordot/ShapeKelz/fc5/Tensordot/ListDiff*
Tparams0*
validate_indices(*#
_output_shapes
:         *
Tindices0
г
Kelz/fc5/Tensordot/Gather_1GatherKelz/fc5/Tensordot/ShapeKelz/fc5/Tensordot/add_1*
Tindices0*
Tparams0*
validate_indices(*
_output_shapes
:
b
Kelz/fc5/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
њ
Kelz/fc5/Tensordot/ProdProdKelz/fc5/Tensordot/GatherKelz/fc5/Tensordot/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
d
Kelz/fc5/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
ў
Kelz/fc5/Tensordot/Prod_1ProdKelz/fc5/Tensordot/Gather_1Kelz/fc5/Tensordot/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
`
Kelz/fc5/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
└
Kelz/fc5/Tensordot/concatConcatV2Kelz/fc5/Tensordot/Gather_1Kelz/fc5/Tensordot/GatherKelz/fc5/Tensordot/concat/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
b
 Kelz/fc5/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
├
Kelz/fc5/Tensordot/concat_1ConcatV2Kelz/fc5/Tensordot/ListDiffKelz/fc5/Tensordot/add_1 Kelz/fc5/Tensordot/concat_1/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
ј
Kelz/fc5/Tensordot/stackPackKelz/fc5/Tensordot/ProdKelz/fc5/Tensordot/Prod_1*
N*
_output_shapes
:*
T0*

axis 
ф
Kelz/fc5/Tensordot/transpose	TransposeKelz/flatten4Kelz/fc5/Tensordot/concat_1*=
_output_shapes+
):'                           *
Tperm0*
T0
д
Kelz/fc5/Tensordot/ReshapeReshapeKelz/fc5/Tensordot/transposeKelz/fc5/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:                  
t
#Kelz/fc5/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
џ
Kelz/fc5/Tensordot/transpose_1	Transposefc5/weights/read#Kelz/fc5/Tensordot/transpose_1/perm*
T0* 
_output_shapes
:
ђ,ђ*
Tperm0
s
"Kelz/fc5/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ц
Kelz/fc5/Tensordot/Reshape_1ReshapeKelz/fc5/Tensordot/transpose_1"Kelz/fc5/Tensordot/Reshape_1/shape*
T0*
Tshape0* 
_output_shapes
:
ђ,ђ
Х
Kelz/fc5/Tensordot/MatMulMatMulKelz/fc5/Tensordot/ReshapeKelz/fc5/Tensordot/Reshape_1*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
e
Kelz/fc5/Tensordot/Const_2Const*
valueB:ђ*
dtype0*
_output_shapes
:
b
 Kelz/fc5/Tensordot/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
├
Kelz/fc5/Tensordot/concat_2ConcatV2Kelz/fc5/Tensordot/GatherKelz/fc5/Tensordot/Const_2 Kelz/fc5/Tensordot/concat_2/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
Б
Kelz/fc5/TensordotReshapeKelz/fc5/Tensordot/MatMulKelz/fc5/Tensordot/concat_2*
T0*
Tshape0*5
_output_shapes#
!:                  ђ
Ќ
Kelz/fc5/BiasAddBiasAddKelz/fc5/Tensordotfc5/biases/read*
T0*
data_formatNHWC*5
_output_shapes#
!:                  ђ
g
Kelz/fc5/ReluReluKelz/fc5/BiasAdd*
T0*5
_output_shapes#
!:                  ђ
d
Kelz/dropout5/dropout/keep_probConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
h
Kelz/dropout5/dropout/ShapeShapeKelz/fc5/Relu*
T0*
out_type0*
_output_shapes
:
m
(Kelz/dropout5/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
(Kelz/dropout5/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
К
2Kelz/dropout5/dropout/random_uniform/RandomUniformRandomUniformKelz/dropout5/dropout/Shape*
T0*
dtype0*5
_output_shapes#
!:                  ђ*
seed2╗*

seed*
ц
(Kelz/dropout5/dropout/random_uniform/subSub(Kelz/dropout5/dropout/random_uniform/max(Kelz/dropout5/dropout/random_uniform/min*
_output_shapes
: *
T0
═
(Kelz/dropout5/dropout/random_uniform/mulMul2Kelz/dropout5/dropout/random_uniform/RandomUniform(Kelz/dropout5/dropout/random_uniform/sub*
T0*5
_output_shapes#
!:                  ђ
┐
$Kelz/dropout5/dropout/random_uniformAdd(Kelz/dropout5/dropout/random_uniform/mul(Kelz/dropout5/dropout/random_uniform/min*
T0*5
_output_shapes#
!:                  ђ
Д
Kelz/dropout5/dropout/addAddKelz/dropout5/dropout/keep_prob$Kelz/dropout5/dropout/random_uniform*
T0*5
_output_shapes#
!:                  ђ

Kelz/dropout5/dropout/FloorFloorKelz/dropout5/dropout/add*
T0*5
_output_shapes#
!:                  ђ
ћ
Kelz/dropout5/dropout/divRealDivKelz/fc5/ReluKelz/dropout5/dropout/keep_prob*
T0*5
_output_shapes#
!:                  ђ
ў
Kelz/dropout5/dropout/mulMulKelz/dropout5/dropout/divKelz/dropout5/dropout/Floor*5
_output_shapes#
!:                  ђ*
T0
Ю
,fc6/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"   X   *
_class
loc:@fc6/weights
Ј
*fc6/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *├лЙ*
_class
loc:@fc6/weights
Ј
*fc6/weights/Initializer/random_uniform/maxConst*
valueB
 *├л>*
_class
loc:@fc6/weights*
dtype0*
_output_shapes
: 
С
4fc6/weights/Initializer/random_uniform/RandomUniformRandomUniform,fc6/weights/Initializer/random_uniform/shape*

seed**
T0*
_class
loc:@fc6/weights*
seed2к*
dtype0*
_output_shapes
:	ђX
╩
*fc6/weights/Initializer/random_uniform/subSub*fc6/weights/Initializer/random_uniform/max*fc6/weights/Initializer/random_uniform/min*
T0*
_class
loc:@fc6/weights*
_output_shapes
: 
П
*fc6/weights/Initializer/random_uniform/mulMul4fc6/weights/Initializer/random_uniform/RandomUniform*fc6/weights/Initializer/random_uniform/sub*
_output_shapes
:	ђX*
T0*
_class
loc:@fc6/weights
¤
&fc6/weights/Initializer/random_uniformAdd*fc6/weights/Initializer/random_uniform/mul*fc6/weights/Initializer/random_uniform/min*
T0*
_class
loc:@fc6/weights*
_output_shapes
:	ђX
А
fc6/weights
VariableV2*
_class
loc:@fc6/weights*
	container *
shape:	ђX*
dtype0*
_output_shapes
:	ђX*
shared_name 
─
fc6/weights/AssignAssignfc6/weights&fc6/weights/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@fc6/weights*
validate_shape(*
_output_shapes
:	ђX
s
fc6/weights/readIdentityfc6/weights*
T0*
_class
loc:@fc6/weights*
_output_shapes
:	ђX
Ћ
,fc6/biases/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:X*
_class
loc:@fc6/biases
є
"fc6/biases/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@fc6/biases*
dtype0*
_output_shapes
: 
╠
fc6/biases/Initializer/zerosFill,fc6/biases/Initializer/zeros/shape_as_tensor"fc6/biases/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@fc6/biases*
_output_shapes
:X
Ћ

fc6/biases
VariableV2*
dtype0*
_output_shapes
:X*
shared_name *
_class
loc:@fc6/biases*
	container *
shape:X
▓
fc6/biases/AssignAssign
fc6/biasesfc6/biases/Initializer/zeros*
T0*
_class
loc:@fc6/biases*
validate_shape(*
_output_shapes
:X*
use_locking(
k
fc6/biases/readIdentity
fc6/biases*
T0*
_class
loc:@fc6/biases*
_output_shapes
:X
q
Kelz/fc6/Tensordot/ShapeShapeKelz/dropout5/dropout/mul*
T0*
out_type0*
_output_shapes
:
Y
Kelz/fc6/Tensordot/RankConst*
value	B :*
dtype0*
_output_shapes
: 
a
Kelz/fc6/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
c
!Kelz/fc6/Tensordot/GreaterEqual/yConst*
dtype0*
_output_shapes
: *
value	B : 
љ
Kelz/fc6/Tensordot/GreaterEqualGreaterEqualKelz/fc6/Tensordot/axes!Kelz/fc6/Tensordot/GreaterEqual/y*
T0*
_output_shapes
:
t
Kelz/fc6/Tensordot/CastCastKelz/fc6/Tensordot/GreaterEqual*

SrcT0
*
_output_shapes
:*

DstT0
t
Kelz/fc6/Tensordot/mulMulKelz/fc6/Tensordot/CastKelz/fc6/Tensordot/axes*
T0*
_output_shapes
:
[
Kelz/fc6/Tensordot/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
x
Kelz/fc6/Tensordot/LessLessKelz/fc6/Tensordot/axesKelz/fc6/Tensordot/Less/y*
T0*
_output_shapes
:
n
Kelz/fc6/Tensordot/Cast_1CastKelz/fc6/Tensordot/Less*
_output_shapes
:*

DstT0*

SrcT0

t
Kelz/fc6/Tensordot/addAddKelz/fc6/Tensordot/axesKelz/fc6/Tensordot/Rank*
_output_shapes
:*
T0
w
Kelz/fc6/Tensordot/mul_1MulKelz/fc6/Tensordot/Cast_1Kelz/fc6/Tensordot/add*
T0*
_output_shapes
:
v
Kelz/fc6/Tensordot/add_1AddKelz/fc6/Tensordot/mulKelz/fc6/Tensordot/mul_1*
T0*
_output_shapes
:
`
Kelz/fc6/Tensordot/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
`
Kelz/fc6/Tensordot/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
б
Kelz/fc6/Tensordot/rangeRangeKelz/fc6/Tensordot/range/startKelz/fc6/Tensordot/RankKelz/fc6/Tensordot/range/delta*
_output_shapes
:*

Tidx0
Д
Kelz/fc6/Tensordot/ListDiffListDiffKelz/fc6/Tensordot/rangeKelz/fc6/Tensordot/add_1*
T0*
out_idx0*2
_output_shapes 
:         :         
Х
Kelz/fc6/Tensordot/GatherGatherKelz/fc6/Tensordot/ShapeKelz/fc6/Tensordot/ListDiff*
Tindices0*
Tparams0*
validate_indices(*#
_output_shapes
:         
г
Kelz/fc6/Tensordot/Gather_1GatherKelz/fc6/Tensordot/ShapeKelz/fc6/Tensordot/add_1*
Tindices0*
Tparams0*
validate_indices(*
_output_shapes
:
b
Kelz/fc6/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
њ
Kelz/fc6/Tensordot/ProdProdKelz/fc6/Tensordot/GatherKelz/fc6/Tensordot/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
d
Kelz/fc6/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
ў
Kelz/fc6/Tensordot/Prod_1ProdKelz/fc6/Tensordot/Gather_1Kelz/fc6/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
`
Kelz/fc6/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
└
Kelz/fc6/Tensordot/concatConcatV2Kelz/fc6/Tensordot/Gather_1Kelz/fc6/Tensordot/GatherKelz/fc6/Tensordot/concat/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
b
 Kelz/fc6/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
├
Kelz/fc6/Tensordot/concat_1ConcatV2Kelz/fc6/Tensordot/ListDiffKelz/fc6/Tensordot/add_1 Kelz/fc6/Tensordot/concat_1/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
ј
Kelz/fc6/Tensordot/stackPackKelz/fc6/Tensordot/ProdKelz/fc6/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
Х
Kelz/fc6/Tensordot/transpose	TransposeKelz/dropout5/dropout/mulKelz/fc6/Tensordot/concat_1*
T0*=
_output_shapes+
):'                           *
Tperm0
д
Kelz/fc6/Tensordot/ReshapeReshapeKelz/fc6/Tensordot/transposeKelz/fc6/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:                  
t
#Kelz/fc6/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       
Ў
Kelz/fc6/Tensordot/transpose_1	Transposefc6/weights/read#Kelz/fc6/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes
:	ђX
s
"Kelz/fc6/Tensordot/Reshape_1/shapeConst*
valueB"   X   *
dtype0*
_output_shapes
:
Б
Kelz/fc6/Tensordot/Reshape_1ReshapeKelz/fc6/Tensordot/transpose_1"Kelz/fc6/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	ђX
х
Kelz/fc6/Tensordot/MatMulMatMulKelz/fc6/Tensordot/ReshapeKelz/fc6/Tensordot/Reshape_1*'
_output_shapes
:         X*
transpose_a( *
transpose_b( *
T0
d
Kelz/fc6/Tensordot/Const_2Const*
valueB:X*
dtype0*
_output_shapes
:
b
 Kelz/fc6/Tensordot/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
├
Kelz/fc6/Tensordot/concat_2ConcatV2Kelz/fc6/Tensordot/GatherKelz/fc6/Tensordot/Const_2 Kelz/fc6/Tensordot/concat_2/axis*

Tidx0*
T0*
N*#
_output_shapes
:         
б
Kelz/fc6/TensordotReshapeKelz/fc6/Tensordot/MatMulKelz/fc6/Tensordot/concat_2*
T0*
Tshape0*4
_output_shapes"
 :                  X
ќ
Kelz/fc6/BiasAddBiasAddKelz/fc6/Tensordotfc6/biases/read*
T0*
data_formatNHWC*4
_output_shapes"
 :                  X
l
Kelz/fc6/SigmoidSigmoidKelz/fc6/BiasAdd*
T0*4
_output_shapes"
 :                  X
S
log_loss/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ћ┐о3
t
log_loss/addAddKelz/fc6/Sigmoidlog_loss/add/y*
T0*4
_output_shapes"
 :                  X
`
log_loss/LogLoglog_loss/add*
T0*4
_output_shapes"
 :                  X
m
log_loss/MulMulPlaceholderlog_loss/Log*
T0*4
_output_shapes"
 :                  X
`
log_loss/NegNeglog_loss/Mul*
T0*4
_output_shapes"
 :                  X
S
log_loss/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
o
log_loss/subSublog_loss/sub/xPlaceholder*4
_output_shapes"
 :                  X*
T0
U
log_loss/sub_1/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
x
log_loss/sub_1Sublog_loss/sub_1/xKelz/fc6/Sigmoid*
T0*4
_output_shapes"
 :                  X
U
log_loss/add_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ћ┐о3
v
log_loss/add_1Addlog_loss/sub_1log_loss/add_1/y*4
_output_shapes"
 :                  X*
T0
d
log_loss/Log_1Loglog_loss/add_1*4
_output_shapes"
 :                  X*
T0
r
log_loss/Mul_1Mullog_loss/sublog_loss/Log_1*4
_output_shapes"
 :                  X*
T0
r
log_loss/sub_2Sublog_loss/Neglog_loss/Mul_1*4
_output_shapes"
 :                  X*
T0
x
+log_loss/assert_broadcastable/weights/shapeShapePlaceholder_1*
_output_shapes
:*
T0*
out_type0
l
*log_loss/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
x
*log_loss/assert_broadcastable/values/shapeShapelog_loss/sub_2*
_output_shapes
:*
T0*
out_type0
k
)log_loss/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
k
)log_loss/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
е
'log_loss/assert_broadcastable/is_scalarEqual)log_loss/assert_broadcastable/is_scalar/x*log_loss/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
▓
3log_loss/assert_broadcastable/is_valid_shape/SwitchSwitch'log_loss/assert_broadcastable/is_scalar'log_loss/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
Ў
5log_loss/assert_broadcastable/is_valid_shape/switch_tIdentity5log_loss/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
Ќ
5log_loss/assert_broadcastable/is_valid_shape/switch_fIdentity3log_loss/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
і
4log_loss/assert_broadcastable/is_valid_shape/pred_idIdentity'log_loss/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
§
5log_loss/assert_broadcastable/is_valid_shape/Switch_1Switch'log_loss/assert_broadcastable/is_scalar4log_loss/assert_broadcastable/is_valid_shape/pred_id*
T0
*:
_class0
.,loc:@log_loss/assert_broadcastable/is_scalar*
_output_shapes
: : 
и
Slog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualZlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch\log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
д
Zlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitch)log_loss/assert_broadcastable/values/rank4log_loss/assert_broadcastable/is_valid_shape/pred_id*
T0*<
_class2
0.loc:@log_loss/assert_broadcastable/values/rank*
_output_shapes
: : 
ф
\log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch*log_loss/assert_broadcastable/weights/rank4log_loss/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0*=
_class3
1/loc:@log_loss/assert_broadcastable/weights/rank
ц
Mlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchSlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankSlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : *
T0

═
Olog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityOlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
╦
Olog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityMlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
л
Nlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentitySlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
Ѓ
flog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
valueB :
         
Ч
blog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsmlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1flog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
┐
ilog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitch*log_loss/assert_broadcastable/values/shape4log_loss/assert_broadcastable/is_valid_shape/pred_id*
T0*=
_class3
1/loc:@log_loss/assert_broadcastable/values/shape* 
_output_shapes
::
џ
klog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switchilog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchNlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*=
_class3
1/loc:@log_loss/assert_broadcastable/values/shape* 
_output_shapes
::
і
glog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
ч
glog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
value	B :
Ш
alog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillglog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shapeglog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*

index_type0*
_output_shapes

:
э
clog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
н
^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2blog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsalog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeclog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
Ё
hlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
ѓ
dlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsolog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1hlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

:
├
klog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitch+log_loss/assert_broadcastable/weights/shape4log_loss/assert_broadcastable/is_valid_shape/pred_id* 
_output_shapes
::*
T0*>
_class4
20loc:@log_loss/assert_broadcastable/weights/shape
Ъ
mlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switchklog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchNlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id* 
_output_shapes
::*
T0*>
_class4
20loc:@log_loss/assert_broadcastable/weights/shape
╔
plog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationdlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*
validate_indices(*<
_output_shapes*
(:         :         :*
set_operationa-b
Ћ
hlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizerlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
ь
Ylog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
к
Wlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualYlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xhlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
_output_shapes
: *
T0
Ѕ
Olog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1SwitchSlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankNlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
_output_shapes
: : *
T0
*f
_class\
ZXloc:@log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank
Ф
Llog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeOlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Wlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
Ь
2log_loss/assert_broadcastable/is_valid_shape/MergeMergeLlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge7log_loss/assert_broadcastable/is_valid_shape/Switch_1:1*
N*
_output_shapes
: : *
T0

І
#log_loss/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
t
%log_loss/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
u
%log_loss/assert_broadcastable/Const_2Const* 
valueB BPlaceholder_1:0*
dtype0*
_output_shapes
: 
s
%log_loss/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
v
%log_loss/assert_broadcastable/Const_4Const*!
valueB Blog_loss/sub_2:0*
dtype0*
_output_shapes
: 
p
%log_loss/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
┼
0log_loss/assert_broadcastable/AssertGuard/SwitchSwitch2log_loss/assert_broadcastable/is_valid_shape/Merge2log_loss/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
Њ
2log_loss/assert_broadcastable/AssertGuard/switch_tIdentity2log_loss/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
Љ
2log_loss/assert_broadcastable/AssertGuard/switch_fIdentity0log_loss/assert_broadcastable/AssertGuard/Switch*
_output_shapes
: *
T0

њ
1log_loss/assert_broadcastable/AssertGuard/pred_idIdentity2log_loss/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
k
.log_loss/assert_broadcastable/AssertGuard/NoOpNoOp3^log_loss/assert_broadcastable/AssertGuard/switch_t
Ћ
<log_loss/assert_broadcastable/AssertGuard/control_dependencyIdentity2log_loss/assert_broadcastable/AssertGuard/switch_t/^log_loss/assert_broadcastable/AssertGuard/NoOp*
T0
*E
_class;
97loc:@log_loss/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
н
7log_loss/assert_broadcastable/AssertGuard/Assert/data_0Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
╗
7log_loss/assert_broadcastable/AssertGuard/Assert/data_1Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
╝
7log_loss/assert_broadcastable/AssertGuard/Assert/data_2Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: * 
valueB BPlaceholder_1:0
║
7log_loss/assert_broadcastable/AssertGuard/Assert/data_4Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB Bvalues.shape=
й
7log_loss/assert_broadcastable/AssertGuard/Assert/data_5Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*!
valueB Blog_loss/sub_2:0*
dtype0*
_output_shapes
: 
и
7log_loss/assert_broadcastable/AssertGuard/Assert/data_7Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Ъ
0log_loss/assert_broadcastable/AssertGuard/AssertAssert7log_loss/assert_broadcastable/AssertGuard/Assert/Switch7log_loss/assert_broadcastable/AssertGuard/Assert/data_07log_loss/assert_broadcastable/AssertGuard/Assert/data_17log_loss/assert_broadcastable/AssertGuard/Assert/data_29log_loss/assert_broadcastable/AssertGuard/Assert/Switch_17log_loss/assert_broadcastable/AssertGuard/Assert/data_47log_loss/assert_broadcastable/AssertGuard/Assert/data_59log_loss/assert_broadcastable/AssertGuard/Assert/Switch_27log_loss/assert_broadcastable/AssertGuard/Assert/data_79log_loss/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
њ
7log_loss/assert_broadcastable/AssertGuard/Assert/SwitchSwitch2log_loss/assert_broadcastable/is_valid_shape/Merge1log_loss/assert_broadcastable/AssertGuard/pred_id*
_output_shapes
: : *
T0
*E
_class;
97loc:@log_loss/assert_broadcastable/is_valid_shape/Merge
ј
9log_loss/assert_broadcastable/AssertGuard/Assert/Switch_1Switch+log_loss/assert_broadcastable/weights/shape1log_loss/assert_broadcastable/AssertGuard/pred_id*
T0*>
_class4
20loc:@log_loss/assert_broadcastable/weights/shape* 
_output_shapes
::
ї
9log_loss/assert_broadcastable/AssertGuard/Assert/Switch_2Switch*log_loss/assert_broadcastable/values/shape1log_loss/assert_broadcastable/AssertGuard/pred_id*
T0*=
_class3
1/loc:@log_loss/assert_broadcastable/values/shape* 
_output_shapes
::
■
9log_loss/assert_broadcastable/AssertGuard/Assert/Switch_3Switch'log_loss/assert_broadcastable/is_scalar1log_loss/assert_broadcastable/AssertGuard/pred_id*
T0
*:
_class0
.,loc:@log_loss/assert_broadcastable/is_scalar*
_output_shapes
: : 
Ў
>log_loss/assert_broadcastable/AssertGuard/control_dependency_1Identity2log_loss/assert_broadcastable/AssertGuard/switch_f1^log_loss/assert_broadcastable/AssertGuard/Assert*
T0
*E
_class;
97loc:@log_loss/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
Р
/log_loss/assert_broadcastable/AssertGuard/MergeMerge>log_loss/assert_broadcastable/AssertGuard/control_dependency_1<log_loss/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
Ц
log_loss/Mul_2Mullog_loss/sub_2Placeholder_10^log_loss/assert_broadcastable/AssertGuard/Merge*
T0*4
_output_shapes"
 :                  X
Ћ
log_loss/ConstConst0^log_loss/assert_broadcastable/AssertGuard/Merge*!
valueB"          *
dtype0*
_output_shapes
:
q
log_loss/SumSumlog_loss/Mul_2log_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Њ
log_loss/num_present/Equal/yConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
valueB
 *    *
dtype0*
_output_shapes
: 
Ј
log_loss/num_present/EqualEqualPlaceholder_1log_loss/num_present/Equal/y*
T0*4
_output_shapes"
 :                  X
г
log_loss/num_present/zeros_like	ZerosLikePlaceholder_10^log_loss/assert_broadcastable/AssertGuard/Merge*
T0*4
_output_shapes"
 :                  X
Б
$log_loss/num_present/ones_like/ShapeShapePlaceholder_10^log_loss/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
Џ
$log_loss/num_present/ones_like/ConstConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
├
log_loss/num_present/ones_likeFill$log_loss/num_present/ones_like/Shape$log_loss/num_present/ones_like/Const*
T0*

index_type0*4
_output_shapes"
 :                  X
┴
log_loss/num_present/SelectSelectlog_loss/num_present/Equallog_loss/num_present/zeros_likelog_loss/num_present/ones_like*4
_output_shapes"
 :                  X*
T0
ц
Ilog_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeShapelog_loss/num_present/Select*
T0*
out_type0*
_output_shapes
:
╝
Hlog_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
value	B :*
dtype0*
_output_shapes
: 
╚
Hlog_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShapelog_loss/sub_20^log_loss/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
╗
Glog_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
value	B :
╗
Glog_loss/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
value	B : *
dtype0*
_output_shapes
: 
ѓ
Elog_loss/num_present/broadcast_weights/assert_broadcastable/is_scalarEqualGlog_loss/num_present/broadcast_weights/assert_broadcastable/is_scalar/xHlog_loss/num_present/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
ї
Qlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchElog_loss/num_present/broadcast_weights/assert_broadcastable/is_scalarElog_loss/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
Н
Slog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentitySlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
М
Slog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentityQlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
_output_shapes
: *
T0

к
Rlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityElog_loss/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
ш
Slog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchElog_loss/num_present/broadcast_weights/assert_broadcastable/is_scalarRlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0
*X
_classN
LJloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
Љ
qlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualxlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchzlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
_output_shapes
: *
T0
ъ
xlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchGlog_loss/num_present/broadcast_weights/assert_broadcastable/values/rankRlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0*Z
_classP
NLloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/values/rank
б
zlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchHlog_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankRlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0*[
_classQ
OMloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/weights/rank
■
klog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchqlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankqlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
Ѕ
mlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitymlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
_output_shapes
: *
T0

Є
mlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityklog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
ї
llog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityqlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
Ы
ёlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst0^log_loss/assert_broadcastable/AssertGuard/Mergen^log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
┘
ђlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsІlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1ёlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
И
Єlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchHlog_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeRlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*[
_classQ
OMloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
ћ
Ѕlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЄlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchllog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*[
_classQ
OMloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
щ
Ёlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst0^log_loss/assert_broadcastable/AssertGuard/Mergen^log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
Ж
Ёlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst0^log_loss/assert_broadcastable/AssertGuard/Mergen^log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
м
log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillЁlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeЁlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
_output_shapes

:*
T0*

index_type0
Т
Ђlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst0^log_loss/assert_broadcastable/AssertGuard/Mergen^log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╬
|log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2ђlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimslog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeЂlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
З
єlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst0^log_loss/assert_broadcastable/AssertGuard/Mergen^log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
▀
ѓlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsЇlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1єlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
_output_shapes

:*

Tdim0*
T0
╝
Ѕlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchIlog_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeRlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id* 
_output_shapes
::*
T0*\
_classR
PNloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape
Ў
Іlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЅlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchllog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*\
_classR
PNloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
Ц
јlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationѓlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1|log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*<
_output_shapes*
(:         :         :*
set_operationa-b*
T0*
validate_indices(
М
єlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeљlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
█
wlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst0^log_loss/assert_broadcastable/AssertGuard/Mergen^log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
value	B : 
А
ulog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualwlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xєlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
ѓ
mlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchqlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankllog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*ё
_classz
xvloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
Ё
jlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergemlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1ulog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
╚
Plog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergejlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeUlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
█
Alog_loss/num_present/broadcast_weights/assert_broadcastable/ConstConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *8
value/B- B'weights can not be broadcast to values.
─
Clog_loss/num_present/broadcast_weights/assert_broadcastable/Const_1Const0^log_loss/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB Bweights.shape=
М
Clog_loss/num_present/broadcast_weights/assert_broadcastable/Const_2Const0^log_loss/assert_broadcastable/AssertGuard/Merge*.
value%B# Blog_loss/num_present/Select:0*
dtype0*
_output_shapes
: 
├
Clog_loss/num_present/broadcast_weights/assert_broadcastable/Const_3Const0^log_loss/assert_broadcastable/AssertGuard/Merge*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
к
Clog_loss/num_present/broadcast_weights/assert_broadcastable/Const_4Const0^log_loss/assert_broadcastable/AssertGuard/Merge*!
valueB Blog_loss/sub_2:0*
dtype0*
_output_shapes
: 
└
Clog_loss/num_present/broadcast_weights/assert_broadcastable/Const_5Const0^log_loss/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB B
is_scalar=
Ъ
Nlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchPlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergePlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : *
T0

¤
Plog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityPlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
_output_shapes
: *
T0

═
Plog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityNlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
_output_shapes
: *
T0

╬
Olog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityPlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
┘
Llog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp0^log_loss/assert_broadcastable/AssertGuard/MergeQ^log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
Ї
Zlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityPlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tM^log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*c
_classY
WUloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
┬
Ulog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const0^log_loss/assert_broadcastable/AssertGuard/MergeQ^log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Е
Ulog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const0^log_loss/assert_broadcastable/AssertGuard/MergeQ^log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB Bweights.shape=
И
Ulog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const0^log_loss/assert_broadcastable/AssertGuard/MergeQ^log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*.
value%B# Blog_loss/num_present/Select:0*
dtype0*
_output_shapes
: 
е
Ulog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const0^log_loss/assert_broadcastable/AssertGuard/MergeQ^log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
Ф
Ulog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const0^log_loss/assert_broadcastable/AssertGuard/MergeQ^log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*!
valueB Blog_loss/sub_2:0*
dtype0*
_output_shapes
: 
Ц
Ulog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const0^log_loss/assert_broadcastable/AssertGuard/MergeQ^log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
ж
Nlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssertUlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchUlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Ulog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Ulog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Wlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1Ulog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Ulog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Wlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2Ulog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Wlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
і
Ulog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchPlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeOlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*c
_classY
WUloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
є
Wlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchIlog_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeOlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*\
_classR
PNloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
ё
Wlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchHlog_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeOlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*[
_classQ
OMloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
Ш
Wlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchElog_loss/num_present/broadcast_weights/assert_broadcastable/is_scalarOlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
_output_shapes
: : *
T0
*X
_classN
LJloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/is_scalar
Љ
\log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityPlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fO^log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*c
_classY
WUloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
╝
Mlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/MergeMerge\log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1Zlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
N*
_output_shapes
: : *
T0

є
6log_loss/num_present/broadcast_weights/ones_like/ShapeShapelog_loss/sub_20^log_loss/assert_broadcastable/AssertGuard/MergeN^log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
§
6log_loss/num_present/broadcast_weights/ones_like/ConstConst0^log_loss/assert_broadcastable/AssertGuard/MergeN^log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
щ
0log_loss/num_present/broadcast_weights/ones_likeFill6log_loss/num_present/broadcast_weights/ones_like/Shape6log_loss/num_present/broadcast_weights/ones_like/Const*4
_output_shapes"
 :                  X*
T0*

index_type0
╗
&log_loss/num_present/broadcast_weightsMullog_loss/num_present/Select0log_loss/num_present/broadcast_weights/ones_like*4
_output_shapes"
 :                  X*
T0
А
log_loss/num_present/ConstConst0^log_loss/assert_broadcastable/AssertGuard/Merge*!
valueB"          *
dtype0*
_output_shapes
:
Ю
log_loss/num_presentSum&log_loss/num_present/broadcast_weightslog_loss/num_present/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ё
log_loss/Const_1Const0^log_loss/assert_broadcastable/AssertGuard/Merge*
valueB *
dtype0*
_output_shapes
: 
s
log_loss/Sum_1Sumlog_loss/Sumlog_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ѕ
log_loss/Greater/yConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB
 *    
f
log_loss/GreaterGreaterlog_loss/num_presentlog_loss/Greater/y*
T0*
_output_shapes
: 
Є
log_loss/Equal/yConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB
 *    
`
log_loss/EqualEquallog_loss/num_presentlog_loss/Equal/y*
T0*
_output_shapes
: 
Ї
log_loss/ones_like/ShapeConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB 
Ј
log_loss/ones_like/ConstConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Ђ
log_loss/ones_likeFilllog_loss/ones_like/Shapelog_loss/ones_like/Const*
_output_shapes
: *
T0*

index_type0
t
log_loss/SelectSelectlog_loss/Equallog_loss/ones_likelog_loss/num_present*
_output_shapes
: *
T0
Y
log_loss/divRealDivlog_loss/Sum_1log_loss/Select*
_output_shapes
: *
T0
ў
#log_loss/zeros_like/shape_as_tensorConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
valueB *
dtype0*
_output_shapes
: 
љ
log_loss/zeros_like/ConstConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
valueB
 *    *
dtype0*
_output_shapes
: 
ј
log_loss/zeros_likeFill#log_loss/zeros_like/shape_as_tensorlog_loss/zeros_like/Const*
T0*

index_type0*
_output_shapes
: 
n
log_loss/valueSelectlog_loss/Greaterlog_loss/divlog_loss/zeros_like*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
{
8gradients/log_loss/value_grad/zeros_like/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
s
.gradients/log_loss/value_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
═
(gradients/log_loss/value_grad/zeros_likeFill8gradients/log_loss/value_grad/zeros_like/shape_as_tensor.gradients/log_loss/value_grad/zeros_like/Const*
T0*

index_type0*
_output_shapes
: 
Џ
$gradients/log_loss/value_grad/SelectSelectlog_loss/Greatergradients/Fill(gradients/log_loss/value_grad/zeros_like*
T0*
_output_shapes
: 
Ю
&gradients/log_loss/value_grad/Select_1Selectlog_loss/Greater(gradients/log_loss/value_grad/zeros_likegradients/Fill*
_output_shapes
: *
T0
є
.gradients/log_loss/value_grad/tuple/group_depsNoOp%^gradients/log_loss/value_grad/Select'^gradients/log_loss/value_grad/Select_1
з
6gradients/log_loss/value_grad/tuple/control_dependencyIdentity$gradients/log_loss/value_grad/Select/^gradients/log_loss/value_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/log_loss/value_grad/Select*
_output_shapes
: 
щ
8gradients/log_loss/value_grad/tuple/control_dependency_1Identity&gradients/log_loss/value_grad/Select_1/^gradients/log_loss/value_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/log_loss/value_grad/Select_1*
_output_shapes
: 
d
!gradients/log_loss/div_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
f
#gradients/log_loss/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
¤
1gradients/log_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/log_loss/div_grad/Shape#gradients/log_loss/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ў
#gradients/log_loss/div_grad/RealDivRealDiv6gradients/log_loss/value_grad/tuple/control_dependencylog_loss/Select*
_output_shapes
: *
T0
Й
gradients/log_loss/div_grad/SumSum#gradients/log_loss/div_grad/RealDiv1gradients/log_loss/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
А
#gradients/log_loss/div_grad/ReshapeReshapegradients/log_loss/div_grad/Sum!gradients/log_loss/div_grad/Shape*
_output_shapes
: *
T0*
Tshape0
W
gradients/log_loss/div_grad/NegNeglog_loss/Sum_1*
_output_shapes
: *
T0
Ѓ
%gradients/log_loss/div_grad/RealDiv_1RealDivgradients/log_loss/div_grad/Neglog_loss/Select*
T0*
_output_shapes
: 
Ѕ
%gradients/log_loss/div_grad/RealDiv_2RealDiv%gradients/log_loss/div_grad/RealDiv_1log_loss/Select*
T0*
_output_shapes
: 
д
gradients/log_loss/div_grad/mulMul6gradients/log_loss/value_grad/tuple/control_dependency%gradients/log_loss/div_grad/RealDiv_2*
T0*
_output_shapes
: 
Й
!gradients/log_loss/div_grad/Sum_1Sumgradients/log_loss/div_grad/mul3gradients/log_loss/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Д
%gradients/log_loss/div_grad/Reshape_1Reshape!gradients/log_loss/div_grad/Sum_1#gradients/log_loss/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ѓ
,gradients/log_loss/div_grad/tuple/group_depsNoOp$^gradients/log_loss/div_grad/Reshape&^gradients/log_loss/div_grad/Reshape_1
ь
4gradients/log_loss/div_grad/tuple/control_dependencyIdentity#gradients/log_loss/div_grad/Reshape-^gradients/log_loss/div_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/log_loss/div_grad/Reshape*
_output_shapes
: 
з
6gradients/log_loss/div_grad/tuple/control_dependency_1Identity%gradients/log_loss/div_grad/Reshape_1-^gradients/log_loss/div_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/div_grad/Reshape_1*
_output_shapes
: 
n
+gradients/log_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
┬
%gradients/log_loss/Sum_1_grad/ReshapeReshape4gradients/log_loss/div_grad/tuple/control_dependency+gradients/log_loss/Sum_1_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0
o
,gradients/log_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
dtype0*
_output_shapes
: 
▓
"gradients/log_loss/Sum_1_grad/TileTile%gradients/log_loss/Sum_1_grad/Reshape,gradients/log_loss/Sum_1_grad/Tile/multiples*
T0*
_output_shapes
: *

Tmultiples0
|
9gradients/log_loss/Select_grad/zeros_like/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
t
/gradients/log_loss/Select_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
л
)gradients/log_loss/Select_grad/zeros_likeFill9gradients/log_loss/Select_grad/zeros_like/shape_as_tensor/gradients/log_loss/Select_grad/zeros_like/Const*
T0*

index_type0*
_output_shapes
: 
├
%gradients/log_loss/Select_grad/SelectSelectlog_loss/Equal6gradients/log_loss/div_grad/tuple/control_dependency_1)gradients/log_loss/Select_grad/zeros_like*
T0*
_output_shapes
: 
┼
'gradients/log_loss/Select_grad/Select_1Selectlog_loss/Equal)gradients/log_loss/Select_grad/zeros_like6gradients/log_loss/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
Ѕ
/gradients/log_loss/Select_grad/tuple/group_depsNoOp&^gradients/log_loss/Select_grad/Select(^gradients/log_loss/Select_grad/Select_1
э
7gradients/log_loss/Select_grad/tuple/control_dependencyIdentity%gradients/log_loss/Select_grad/Select0^gradients/log_loss/Select_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/Select_grad/Select*
_output_shapes
: 
§
9gradients/log_loss/Select_grad/tuple/control_dependency_1Identity'gradients/log_loss/Select_grad/Select_10^gradients/log_loss/Select_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/log_loss/Select_grad/Select_1*
_output_shapes
: 
~
)gradients/log_loss/Sum_grad/Reshape/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
И
#gradients/log_loss/Sum_grad/ReshapeReshape"gradients/log_loss/Sum_1_grad/Tile)gradients/log_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*"
_output_shapes
:
o
!gradients/log_loss/Sum_grad/ShapeShapelog_loss/Mul_2*
T0*
out_type0*
_output_shapes
:
┴
 gradients/log_loss/Sum_grad/TileTile#gradients/log_loss/Sum_grad/Reshape!gradients/log_loss/Sum_grad/Shape*

Tmultiples0*
T0*4
_output_shapes"
 :                  X
q
#gradients/log_loss/Mul_2_grad/ShapeShapelog_loss/sub_2*
_output_shapes
:*
T0*
out_type0
r
%gradients/log_loss/Mul_2_grad/Shape_1ShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
Н
3gradients/log_loss/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/log_loss/Mul_2_grad/Shape%gradients/log_loss/Mul_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ў
!gradients/log_loss/Mul_2_grad/mulMul gradients/log_loss/Sum_grad/TilePlaceholder_1*
T0*4
_output_shapes"
 :                  X
└
!gradients/log_loss/Mul_2_grad/SumSum!gradients/log_loss/Mul_2_grad/mul3gradients/log_loss/Mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
┼
%gradients/log_loss/Mul_2_grad/ReshapeReshape!gradients/log_loss/Mul_2_grad/Sum#gradients/log_loss/Mul_2_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :                  X
Џ
#gradients/log_loss/Mul_2_grad/mul_1Mullog_loss/sub_2 gradients/log_loss/Sum_grad/Tile*
T0*4
_output_shapes"
 :                  X
к
#gradients/log_loss/Mul_2_grad/Sum_1Sum#gradients/log_loss/Mul_2_grad/mul_15gradients/log_loss/Mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
╦
'gradients/log_loss/Mul_2_grad/Reshape_1Reshape#gradients/log_loss/Mul_2_grad/Sum_1%gradients/log_loss/Mul_2_grad/Shape_1*
T0*
Tshape0*4
_output_shapes"
 :                  X
ѕ
.gradients/log_loss/Mul_2_grad/tuple/group_depsNoOp&^gradients/log_loss/Mul_2_grad/Reshape(^gradients/log_loss/Mul_2_grad/Reshape_1
Њ
6gradients/log_loss/Mul_2_grad/tuple/control_dependencyIdentity%gradients/log_loss/Mul_2_grad/Reshape/^gradients/log_loss/Mul_2_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/Mul_2_grad/Reshape*4
_output_shapes"
 :                  X
Ў
8gradients/log_loss/Mul_2_grad/tuple/control_dependency_1Identity'gradients/log_loss/Mul_2_grad/Reshape_1/^gradients/log_loss/Mul_2_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/log_loss/Mul_2_grad/Reshape_1*4
_output_shapes"
 :                  X
є
1gradients/log_loss/num_present_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*!
valueB"         
▀
+gradients/log_loss/num_present_grad/ReshapeReshape9gradients/log_loss/Select_grad/tuple/control_dependency_11gradients/log_loss/num_present_grad/Reshape/shape*
T0*
Tshape0*"
_output_shapes
:
Ј
)gradients/log_loss/num_present_grad/ShapeShape&log_loss/num_present/broadcast_weights*
T0*
out_type0*
_output_shapes
:
┘
(gradients/log_loss/num_present_grad/TileTile+gradients/log_loss/num_present_grad/Reshape)gradients/log_loss/num_present_grad/Shape*
T0*4
_output_shapes"
 :                  X*

Tmultiples0
ќ
;gradients/log_loss/num_present/broadcast_weights_grad/ShapeShapelog_loss/num_present/Select*
_output_shapes
:*
T0*
out_type0
Г
=gradients/log_loss/num_present/broadcast_weights_grad/Shape_1Shape0log_loss/num_present/broadcast_weights/ones_like*
_output_shapes
:*
T0*
out_type0
Ю
Kgradients/log_loss/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/log_loss/num_present/broadcast_weights_grad/Shape=gradients/log_loss/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:         :         
█
9gradients/log_loss/num_present/broadcast_weights_grad/mulMul(gradients/log_loss/num_present_grad/Tile0log_loss/num_present/broadcast_weights/ones_like*
T0*4
_output_shapes"
 :                  X
ѕ
9gradients/log_loss/num_present/broadcast_weights_grad/SumSum9gradients/log_loss/num_present/broadcast_weights_grad/mulKgradients/log_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ї
=gradients/log_loss/num_present/broadcast_weights_grad/ReshapeReshape9gradients/log_loss/num_present/broadcast_weights_grad/Sum;gradients/log_loss/num_present/broadcast_weights_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :                  X
╚
;gradients/log_loss/num_present/broadcast_weights_grad/mul_1Mullog_loss/num_present/Select(gradients/log_loss/num_present_grad/Tile*
T0*4
_output_shapes"
 :                  X
ј
;gradients/log_loss/num_present/broadcast_weights_grad/Sum_1Sum;gradients/log_loss/num_present/broadcast_weights_grad/mul_1Mgradients/log_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Њ
?gradients/log_loss/num_present/broadcast_weights_grad/Reshape_1Reshape;gradients/log_loss/num_present/broadcast_weights_grad/Sum_1=gradients/log_loss/num_present/broadcast_weights_grad/Shape_1*
T0*
Tshape0*4
_output_shapes"
 :                  X
л
Fgradients/log_loss/num_present/broadcast_weights_grad/tuple/group_depsNoOp>^gradients/log_loss/num_present/broadcast_weights_grad/Reshape@^gradients/log_loss/num_present/broadcast_weights_grad/Reshape_1
з
Ngradients/log_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentity=gradients/log_loss/num_present/broadcast_weights_grad/ReshapeG^gradients/log_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/log_loss/num_present/broadcast_weights_grad/Reshape*4
_output_shapes"
 :                  X
щ
Pgradients/log_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Identity?gradients/log_loss/num_present/broadcast_weights_grad/Reshape_1G^gradients/log_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/log_loss/num_present/broadcast_weights_grad/Reshape_1*4
_output_shapes"
 :                  X
џ
Egradients/log_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*
dtype0*
_output_shapes
:*!
valueB"          
А
Cgradients/log_loss/num_present/broadcast_weights/ones_like_grad/SumSumPgradients/log_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Egradients/log_loss/num_present/broadcast_weights/ones_like_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
o
#gradients/log_loss/sub_2_grad/ShapeShapelog_loss/Neg*
T0*
out_type0*
_output_shapes
:
s
%gradients/log_loss/sub_2_grad/Shape_1Shapelog_loss/Mul_1*
T0*
out_type0*
_output_shapes
:
Н
3gradients/log_loss/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/log_loss/sub_2_grad/Shape%gradients/log_loss/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Н
!gradients/log_loss/sub_2_grad/SumSum6gradients/log_loss/Mul_2_grad/tuple/control_dependency3gradients/log_loss/sub_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
┼
%gradients/log_loss/sub_2_grad/ReshapeReshape!gradients/log_loss/sub_2_grad/Sum#gradients/log_loss/sub_2_grad/Shape*4
_output_shapes"
 :                  X*
T0*
Tshape0
┘
#gradients/log_loss/sub_2_grad/Sum_1Sum6gradients/log_loss/Mul_2_grad/tuple/control_dependency5gradients/log_loss/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
p
!gradients/log_loss/sub_2_grad/NegNeg#gradients/log_loss/sub_2_grad/Sum_1*
T0*
_output_shapes
:
╔
'gradients/log_loss/sub_2_grad/Reshape_1Reshape!gradients/log_loss/sub_2_grad/Neg%gradients/log_loss/sub_2_grad/Shape_1*
T0*
Tshape0*4
_output_shapes"
 :                  X
ѕ
.gradients/log_loss/sub_2_grad/tuple/group_depsNoOp&^gradients/log_loss/sub_2_grad/Reshape(^gradients/log_loss/sub_2_grad/Reshape_1
Њ
6gradients/log_loss/sub_2_grad/tuple/control_dependencyIdentity%gradients/log_loss/sub_2_grad/Reshape/^gradients/log_loss/sub_2_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/sub_2_grad/Reshape*4
_output_shapes"
 :                  X
Ў
8gradients/log_loss/sub_2_grad/tuple/control_dependency_1Identity'gradients/log_loss/sub_2_grad/Reshape_1/^gradients/log_loss/sub_2_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/log_loss/sub_2_grad/Reshape_1*4
_output_shapes"
 :                  X
Ю
gradients/log_loss/Neg_grad/NegNeg6gradients/log_loss/sub_2_grad/tuple/control_dependency*4
_output_shapes"
 :                  X*
T0
o
#gradients/log_loss/Mul_1_grad/ShapeShapelog_loss/sub*
T0*
out_type0*
_output_shapes
:
s
%gradients/log_loss/Mul_1_grad/Shape_1Shapelog_loss/Log_1*
T0*
out_type0*
_output_shapes
:
Н
3gradients/log_loss/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/log_loss/Mul_1_grad/Shape%gradients/log_loss/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
▒
!gradients/log_loss/Mul_1_grad/mulMul8gradients/log_loss/sub_2_grad/tuple/control_dependency_1log_loss/Log_1*
T0*4
_output_shapes"
 :                  X
└
!gradients/log_loss/Mul_1_grad/SumSum!gradients/log_loss/Mul_1_grad/mul3gradients/log_loss/Mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
┼
%gradients/log_loss/Mul_1_grad/ReshapeReshape!gradients/log_loss/Mul_1_grad/Sum#gradients/log_loss/Mul_1_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :                  X
▒
#gradients/log_loss/Mul_1_grad/mul_1Mullog_loss/sub8gradients/log_loss/sub_2_grad/tuple/control_dependency_1*
T0*4
_output_shapes"
 :                  X
к
#gradients/log_loss/Mul_1_grad/Sum_1Sum#gradients/log_loss/Mul_1_grad/mul_15gradients/log_loss/Mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
╦
'gradients/log_loss/Mul_1_grad/Reshape_1Reshape#gradients/log_loss/Mul_1_grad/Sum_1%gradients/log_loss/Mul_1_grad/Shape_1*4
_output_shapes"
 :                  X*
T0*
Tshape0
ѕ
.gradients/log_loss/Mul_1_grad/tuple/group_depsNoOp&^gradients/log_loss/Mul_1_grad/Reshape(^gradients/log_loss/Mul_1_grad/Reshape_1
Њ
6gradients/log_loss/Mul_1_grad/tuple/control_dependencyIdentity%gradients/log_loss/Mul_1_grad/Reshape/^gradients/log_loss/Mul_1_grad/tuple/group_deps*4
_output_shapes"
 :                  X*
T0*8
_class.
,*loc:@gradients/log_loss/Mul_1_grad/Reshape
Ў
8gradients/log_loss/Mul_1_grad/tuple/control_dependency_1Identity'gradients/log_loss/Mul_1_grad/Reshape_1/^gradients/log_loss/Mul_1_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/log_loss/Mul_1_grad/Reshape_1*4
_output_shapes"
 :                  X
l
!gradients/log_loss/Mul_grad/ShapeShapePlaceholder*
_output_shapes
:*
T0*
out_type0
o
#gradients/log_loss/Mul_grad/Shape_1Shapelog_loss/Log*
T0*
out_type0*
_output_shapes
:
¤
1gradients/log_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/log_loss/Mul_grad/Shape#gradients/log_loss/Mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ћ
gradients/log_loss/Mul_grad/mulMulgradients/log_loss/Neg_grad/Neglog_loss/Log*4
_output_shapes"
 :                  X*
T0
║
gradients/log_loss/Mul_grad/SumSumgradients/log_loss/Mul_grad/mul1gradients/log_loss/Mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
┐
#gradients/log_loss/Mul_grad/ReshapeReshapegradients/log_loss/Mul_grad/Sum!gradients/log_loss/Mul_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :                  X
Ћ
!gradients/log_loss/Mul_grad/mul_1MulPlaceholdergradients/log_loss/Neg_grad/Neg*
T0*4
_output_shapes"
 :                  X
└
!gradients/log_loss/Mul_grad/Sum_1Sum!gradients/log_loss/Mul_grad/mul_13gradients/log_loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
┼
%gradients/log_loss/Mul_grad/Reshape_1Reshape!gradients/log_loss/Mul_grad/Sum_1#gradients/log_loss/Mul_grad/Shape_1*
T0*
Tshape0*4
_output_shapes"
 :                  X
ѓ
,gradients/log_loss/Mul_grad/tuple/group_depsNoOp$^gradients/log_loss/Mul_grad/Reshape&^gradients/log_loss/Mul_grad/Reshape_1
І
4gradients/log_loss/Mul_grad/tuple/control_dependencyIdentity#gradients/log_loss/Mul_grad/Reshape-^gradients/log_loss/Mul_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/log_loss/Mul_grad/Reshape*4
_output_shapes"
 :                  X
Љ
6gradients/log_loss/Mul_grad/tuple/control_dependency_1Identity%gradients/log_loss/Mul_grad/Reshape_1-^gradients/log_loss/Mul_grad/tuple/group_deps*4
_output_shapes"
 :                  X*
T0*8
_class.
,*loc:@gradients/log_loss/Mul_grad/Reshape_1
└
(gradients/log_loss/Log_1_grad/Reciprocal
Reciprocallog_loss/add_19^gradients/log_loss/Mul_1_grad/tuple/control_dependency_1*
T0*4
_output_shapes"
 :                  X
╦
!gradients/log_loss/Log_1_grad/mulMul8gradients/log_loss/Mul_1_grad/tuple/control_dependency_1(gradients/log_loss/Log_1_grad/Reciprocal*4
_output_shapes"
 :                  X*
T0
║
&gradients/log_loss/Log_grad/Reciprocal
Reciprocallog_loss/add7^gradients/log_loss/Mul_grad/tuple/control_dependency_1*
T0*4
_output_shapes"
 :                  X
┼
gradients/log_loss/Log_grad/mulMul6gradients/log_loss/Mul_grad/tuple/control_dependency_1&gradients/log_loss/Log_grad/Reciprocal*4
_output_shapes"
 :                  X*
T0
q
#gradients/log_loss/add_1_grad/ShapeShapelog_loss/sub_1*
T0*
out_type0*
_output_shapes
:
h
%gradients/log_loss/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
Н
3gradients/log_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/log_loss/add_1_grad/Shape%gradients/log_loss/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
└
!gradients/log_loss/add_1_grad/SumSum!gradients/log_loss/Log_1_grad/mul3gradients/log_loss/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
┼
%gradients/log_loss/add_1_grad/ReshapeReshape!gradients/log_loss/add_1_grad/Sum#gradients/log_loss/add_1_grad/Shape*4
_output_shapes"
 :                  X*
T0*
Tshape0
─
#gradients/log_loss/add_1_grad/Sum_1Sum!gradients/log_loss/Log_1_grad/mul5gradients/log_loss/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Г
'gradients/log_loss/add_1_grad/Reshape_1Reshape#gradients/log_loss/add_1_grad/Sum_1%gradients/log_loss/add_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
ѕ
.gradients/log_loss/add_1_grad/tuple/group_depsNoOp&^gradients/log_loss/add_1_grad/Reshape(^gradients/log_loss/add_1_grad/Reshape_1
Њ
6gradients/log_loss/add_1_grad/tuple/control_dependencyIdentity%gradients/log_loss/add_1_grad/Reshape/^gradients/log_loss/add_1_grad/tuple/group_deps*4
_output_shapes"
 :                  X*
T0*8
_class.
,*loc:@gradients/log_loss/add_1_grad/Reshape
ч
8gradients/log_loss/add_1_grad/tuple/control_dependency_1Identity'gradients/log_loss/add_1_grad/Reshape_1/^gradients/log_loss/add_1_grad/tuple/group_deps*
_output_shapes
: *
T0*:
_class0
.,loc:@gradients/log_loss/add_1_grad/Reshape_1
q
!gradients/log_loss/add_grad/ShapeShapeKelz/fc6/Sigmoid*
T0*
out_type0*
_output_shapes
:
f
#gradients/log_loss/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
¤
1gradients/log_loss/add_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/log_loss/add_grad/Shape#gradients/log_loss/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
║
gradients/log_loss/add_grad/SumSumgradients/log_loss/Log_grad/mul1gradients/log_loss/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
┐
#gradients/log_loss/add_grad/ReshapeReshapegradients/log_loss/add_grad/Sum!gradients/log_loss/add_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :                  X
Й
!gradients/log_loss/add_grad/Sum_1Sumgradients/log_loss/Log_grad/mul3gradients/log_loss/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Д
%gradients/log_loss/add_grad/Reshape_1Reshape!gradients/log_loss/add_grad/Sum_1#gradients/log_loss/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ѓ
,gradients/log_loss/add_grad/tuple/group_depsNoOp$^gradients/log_loss/add_grad/Reshape&^gradients/log_loss/add_grad/Reshape_1
І
4gradients/log_loss/add_grad/tuple/control_dependencyIdentity#gradients/log_loss/add_grad/Reshape-^gradients/log_loss/add_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/log_loss/add_grad/Reshape*4
_output_shapes"
 :                  X
з
6gradients/log_loss/add_grad/tuple/control_dependency_1Identity%gradients/log_loss/add_grad/Reshape_1-^gradients/log_loss/add_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/add_grad/Reshape_1*
_output_shapes
: 
f
#gradients/log_loss/sub_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
u
%gradients/log_loss/sub_1_grad/Shape_1ShapeKelz/fc6/Sigmoid*
T0*
out_type0*
_output_shapes
:
Н
3gradients/log_loss/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/log_loss/sub_1_grad/Shape%gradients/log_loss/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Н
!gradients/log_loss/sub_1_grad/SumSum6gradients/log_loss/add_1_grad/tuple/control_dependency3gradients/log_loss/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Д
%gradients/log_loss/sub_1_grad/ReshapeReshape!gradients/log_loss/sub_1_grad/Sum#gradients/log_loss/sub_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
┘
#gradients/log_loss/sub_1_grad/Sum_1Sum6gradients/log_loss/add_1_grad/tuple/control_dependency5gradients/log_loss/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
p
!gradients/log_loss/sub_1_grad/NegNeg#gradients/log_loss/sub_1_grad/Sum_1*
T0*
_output_shapes
:
╔
'gradients/log_loss/sub_1_grad/Reshape_1Reshape!gradients/log_loss/sub_1_grad/Neg%gradients/log_loss/sub_1_grad/Shape_1*
T0*
Tshape0*4
_output_shapes"
 :                  X
ѕ
.gradients/log_loss/sub_1_grad/tuple/group_depsNoOp&^gradients/log_loss/sub_1_grad/Reshape(^gradients/log_loss/sub_1_grad/Reshape_1
ш
6gradients/log_loss/sub_1_grad/tuple/control_dependencyIdentity%gradients/log_loss/sub_1_grad/Reshape/^gradients/log_loss/sub_1_grad/tuple/group_deps*
_output_shapes
: *
T0*8
_class.
,*loc:@gradients/log_loss/sub_1_grad/Reshape
Ў
8gradients/log_loss/sub_1_grad/tuple/control_dependency_1Identity'gradients/log_loss/sub_1_grad/Reshape_1/^gradients/log_loss/sub_1_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/log_loss/sub_1_grad/Reshape_1*4
_output_shapes"
 :                  X
є
gradients/AddNAddN4gradients/log_loss/add_grad/tuple/control_dependency8gradients/log_loss/sub_1_grad/tuple/control_dependency_1*
T0*6
_class,
*(loc:@gradients/log_loss/add_grad/Reshape*
N*4
_output_shapes"
 :                  X
Џ
+gradients/Kelz/fc6/Sigmoid_grad/SigmoidGradSigmoidGradKelz/fc6/Sigmoidgradients/AddN*
T0*4
_output_shapes"
 :                  X
Б
+gradients/Kelz/fc6/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients/Kelz/fc6/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:X*
T0
ћ
0gradients/Kelz/fc6/BiasAdd_grad/tuple/group_depsNoOp,^gradients/Kelz/fc6/Sigmoid_grad/SigmoidGrad,^gradients/Kelz/fc6/BiasAdd_grad/BiasAddGrad
Б
8gradients/Kelz/fc6/BiasAdd_grad/tuple/control_dependencyIdentity+gradients/Kelz/fc6/Sigmoid_grad/SigmoidGrad1^gradients/Kelz/fc6/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Kelz/fc6/Sigmoid_grad/SigmoidGrad*4
_output_shapes"
 :                  X
І
:gradients/Kelz/fc6/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/Kelz/fc6/BiasAdd_grad/BiasAddGrad1^gradients/Kelz/fc6/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Kelz/fc6/BiasAdd_grad/BiasAddGrad*
_output_shapes
:X
ђ
'gradients/Kelz/fc6/Tensordot_grad/ShapeShapeKelz/fc6/Tensordot/MatMul*
_output_shapes
:*
T0*
out_type0
О
)gradients/Kelz/fc6/Tensordot_grad/ReshapeReshape8gradients/Kelz/fc6/BiasAdd_grad/tuple/control_dependency'gradients/Kelz/fc6/Tensordot_grad/Shape*'
_output_shapes
:         X*
T0*
Tshape0
█
/gradients/Kelz/fc6/Tensordot/MatMul_grad/MatMulMatMul)gradients/Kelz/fc6/Tensordot_grad/ReshapeKelz/fc6/Tensordot/Reshape_1*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
┌
1gradients/Kelz/fc6/Tensordot/MatMul_grad/MatMul_1MatMulKelz/fc6/Tensordot/Reshape)gradients/Kelz/fc6/Tensordot_grad/Reshape*
T0*'
_output_shapes
:         X*
transpose_a(*
transpose_b( 
Д
9gradients/Kelz/fc6/Tensordot/MatMul_grad/tuple/group_depsNoOp0^gradients/Kelz/fc6/Tensordot/MatMul_grad/MatMul2^gradients/Kelz/fc6/Tensordot/MatMul_grad/MatMul_1
▒
Agradients/Kelz/fc6/Tensordot/MatMul_grad/tuple/control_dependencyIdentity/gradients/Kelz/fc6/Tensordot/MatMul_grad/MatMul:^gradients/Kelz/fc6/Tensordot/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*B
_class8
64loc:@gradients/Kelz/fc6/Tensordot/MatMul_grad/MatMul
«
Cgradients/Kelz/fc6/Tensordot/MatMul_grad/tuple/control_dependency_1Identity1gradients/Kelz/fc6/Tensordot/MatMul_grad/MatMul_1:^gradients/Kelz/fc6/Tensordot/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Kelz/fc6/Tensordot/MatMul_grad/MatMul_1*
_output_shapes
:	ђX
І
/gradients/Kelz/fc6/Tensordot/Reshape_grad/ShapeShapeKelz/fc6/Tensordot/transpose*
_output_shapes
:*
T0*
out_type0
є
1gradients/Kelz/fc6/Tensordot/Reshape_grad/ReshapeReshapeAgradients/Kelz/fc6/Tensordot/MatMul_grad/tuple/control_dependency/gradients/Kelz/fc6/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*=
_output_shapes+
):'                           
ѓ
1gradients/Kelz/fc6/Tensordot/Reshape_1_grad/ShapeConst*
valueB"   X   *
dtype0*
_output_shapes
:
Ь
3gradients/Kelz/fc6/Tensordot/Reshape_1_grad/ReshapeReshapeCgradients/Kelz/fc6/Tensordot/MatMul_grad/tuple/control_dependency_11gradients/Kelz/fc6/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:	ђX
Ю
=gradients/Kelz/fc6/Tensordot/transpose_grad/InvertPermutationInvertPermutationKelz/fc6/Tensordot/concat_1*#
_output_shapes
:         *
T0
Ђ
5gradients/Kelz/fc6/Tensordot/transpose_grad/transpose	Transpose1gradients/Kelz/fc6/Tensordot/Reshape_grad/Reshape=gradients/Kelz/fc6/Tensordot/transpose_grad/InvertPermutation*5
_output_shapes#
!:                  ђ*
Tperm0*
T0
ъ
?gradients/Kelz/fc6/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation#Kelz/fc6/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
ы
7gradients/Kelz/fc6/Tensordot/transpose_1_grad/transpose	Transpose3gradients/Kelz/fc6/Tensordot/Reshape_1_grad/Reshape?gradients/Kelz/fc6/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes
:	ђX
Є
.gradients/Kelz/dropout5/dropout/mul_grad/ShapeShapeKelz/dropout5/dropout/div*
T0*
out_type0*
_output_shapes
:
І
0gradients/Kelz/dropout5/dropout/mul_grad/Shape_1ShapeKelz/dropout5/dropout/Floor*
T0*
out_type0*
_output_shapes
:
Ш
>gradients/Kelz/dropout5/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/Kelz/dropout5/dropout/mul_grad/Shape0gradients/Kelz/dropout5/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
К
,gradients/Kelz/dropout5/dropout/mul_grad/mulMul5gradients/Kelz/fc6/Tensordot/transpose_grad/transposeKelz/dropout5/dropout/Floor*
T0*5
_output_shapes#
!:                  ђ
р
,gradients/Kelz/dropout5/dropout/mul_grad/SumSum,gradients/Kelz/dropout5/dropout/mul_grad/mul>gradients/Kelz/dropout5/dropout/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
у
0gradients/Kelz/dropout5/dropout/mul_grad/ReshapeReshape,gradients/Kelz/dropout5/dropout/mul_grad/Sum.gradients/Kelz/dropout5/dropout/mul_grad/Shape*
T0*
Tshape0*5
_output_shapes#
!:                  ђ
К
.gradients/Kelz/dropout5/dropout/mul_grad/mul_1MulKelz/dropout5/dropout/div5gradients/Kelz/fc6/Tensordot/transpose_grad/transpose*5
_output_shapes#
!:                  ђ*
T0
у
.gradients/Kelz/dropout5/dropout/mul_grad/Sum_1Sum.gradients/Kelz/dropout5/dropout/mul_grad/mul_1@gradients/Kelz/dropout5/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ь
2gradients/Kelz/dropout5/dropout/mul_grad/Reshape_1Reshape.gradients/Kelz/dropout5/dropout/mul_grad/Sum_10gradients/Kelz/dropout5/dropout/mul_grad/Shape_1*
T0*
Tshape0*5
_output_shapes#
!:                  ђ
Е
9gradients/Kelz/dropout5/dropout/mul_grad/tuple/group_depsNoOp1^gradients/Kelz/dropout5/dropout/mul_grad/Reshape3^gradients/Kelz/dropout5/dropout/mul_grad/Reshape_1
└
Agradients/Kelz/dropout5/dropout/mul_grad/tuple/control_dependencyIdentity0gradients/Kelz/dropout5/dropout/mul_grad/Reshape:^gradients/Kelz/dropout5/dropout/mul_grad/tuple/group_deps*5
_output_shapes#
!:                  ђ*
T0*C
_class9
75loc:@gradients/Kelz/dropout5/dropout/mul_grad/Reshape
к
Cgradients/Kelz/dropout5/dropout/mul_grad/tuple/control_dependency_1Identity2gradients/Kelz/dropout5/dropout/mul_grad/Reshape_1:^gradients/Kelz/dropout5/dropout/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/Kelz/dropout5/dropout/mul_grad/Reshape_1*5
_output_shapes#
!:                  ђ
{
.gradients/Kelz/dropout5/dropout/div_grad/ShapeShapeKelz/fc5/Relu*
T0*
out_type0*
_output_shapes
:
s
0gradients/Kelz/dropout5/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ш
>gradients/Kelz/dropout5/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/Kelz/dropout5/dropout/div_grad/Shape0gradients/Kelz/dropout5/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
▀
0gradients/Kelz/dropout5/dropout/div_grad/RealDivRealDivAgradients/Kelz/dropout5/dropout/mul_grad/tuple/control_dependencyKelz/dropout5/dropout/keep_prob*
T0*5
_output_shapes#
!:                  ђ
т
,gradients/Kelz/dropout5/dropout/div_grad/SumSum0gradients/Kelz/dropout5/dropout/div_grad/RealDiv>gradients/Kelz/dropout5/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
у
0gradients/Kelz/dropout5/dropout/div_grad/ReshapeReshape,gradients/Kelz/dropout5/dropout/div_grad/Sum.gradients/Kelz/dropout5/dropout/div_grad/Shape*
T0*
Tshape0*5
_output_shapes#
!:                  ђ
ѓ
,gradients/Kelz/dropout5/dropout/div_grad/NegNegKelz/fc5/Relu*5
_output_shapes#
!:                  ђ*
T0
╠
2gradients/Kelz/dropout5/dropout/div_grad/RealDiv_1RealDiv,gradients/Kelz/dropout5/dropout/div_grad/NegKelz/dropout5/dropout/keep_prob*
T0*5
_output_shapes#
!:                  ђ
м
2gradients/Kelz/dropout5/dropout/div_grad/RealDiv_2RealDiv2gradients/Kelz/dropout5/dropout/div_grad/RealDiv_1Kelz/dropout5/dropout/keep_prob*5
_output_shapes#
!:                  ђ*
T0
Ж
,gradients/Kelz/dropout5/dropout/div_grad/mulMulAgradients/Kelz/dropout5/dropout/mul_grad/tuple/control_dependency2gradients/Kelz/dropout5/dropout/div_grad/RealDiv_2*5
_output_shapes#
!:                  ђ*
T0
т
.gradients/Kelz/dropout5/dropout/div_grad/Sum_1Sum,gradients/Kelz/dropout5/dropout/div_grad/mul@gradients/Kelz/dropout5/dropout/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
╬
2gradients/Kelz/dropout5/dropout/div_grad/Reshape_1Reshape.gradients/Kelz/dropout5/dropout/div_grad/Sum_10gradients/Kelz/dropout5/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Е
9gradients/Kelz/dropout5/dropout/div_grad/tuple/group_depsNoOp1^gradients/Kelz/dropout5/dropout/div_grad/Reshape3^gradients/Kelz/dropout5/dropout/div_grad/Reshape_1
└
Agradients/Kelz/dropout5/dropout/div_grad/tuple/control_dependencyIdentity0gradients/Kelz/dropout5/dropout/div_grad/Reshape:^gradients/Kelz/dropout5/dropout/div_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Kelz/dropout5/dropout/div_grad/Reshape*5
_output_shapes#
!:                  ђ
Д
Cgradients/Kelz/dropout5/dropout/div_grad/tuple/control_dependency_1Identity2gradients/Kelz/dropout5/dropout/div_grad/Reshape_1:^gradients/Kelz/dropout5/dropout/div_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/Kelz/dropout5/dropout/div_grad/Reshape_1*
_output_shapes
: 
├
%gradients/Kelz/fc5/Relu_grad/ReluGradReluGradAgradients/Kelz/dropout5/dropout/div_grad/tuple/control_dependencyKelz/fc5/Relu*5
_output_shapes#
!:                  ђ*
T0
ъ
+gradients/Kelz/fc5/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/Kelz/fc5/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:ђ
ј
0gradients/Kelz/fc5/BiasAdd_grad/tuple/group_depsNoOp&^gradients/Kelz/fc5/Relu_grad/ReluGrad,^gradients/Kelz/fc5/BiasAdd_grad/BiasAddGrad
ў
8gradients/Kelz/fc5/BiasAdd_grad/tuple/control_dependencyIdentity%gradients/Kelz/fc5/Relu_grad/ReluGrad1^gradients/Kelz/fc5/BiasAdd_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/Kelz/fc5/Relu_grad/ReluGrad*5
_output_shapes#
!:                  ђ
ї
:gradients/Kelz/fc5/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/Kelz/fc5/BiasAdd_grad/BiasAddGrad1^gradients/Kelz/fc5/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Kelz/fc5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
ђ
'gradients/Kelz/fc5/Tensordot_grad/ShapeShapeKelz/fc5/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
п
)gradients/Kelz/fc5/Tensordot_grad/ReshapeReshape8gradients/Kelz/fc5/BiasAdd_grad/tuple/control_dependency'gradients/Kelz/fc5/Tensordot_grad/Shape*(
_output_shapes
:         ђ*
T0*
Tshape0
█
/gradients/Kelz/fc5/Tensordot/MatMul_grad/MatMulMatMul)gradients/Kelz/fc5/Tensordot_grad/ReshapeKelz/fc5/Tensordot/Reshape_1*(
_output_shapes
:         ђ,*
transpose_a( *
transpose_b(*
T0
█
1gradients/Kelz/fc5/Tensordot/MatMul_grad/MatMul_1MatMulKelz/fc5/Tensordot/Reshape)gradients/Kelz/fc5/Tensordot_grad/Reshape*
T0*(
_output_shapes
:         ђ*
transpose_a(*
transpose_b( 
Д
9gradients/Kelz/fc5/Tensordot/MatMul_grad/tuple/group_depsNoOp0^gradients/Kelz/fc5/Tensordot/MatMul_grad/MatMul2^gradients/Kelz/fc5/Tensordot/MatMul_grad/MatMul_1
▒
Agradients/Kelz/fc5/Tensordot/MatMul_grad/tuple/control_dependencyIdentity/gradients/Kelz/fc5/Tensordot/MatMul_grad/MatMul:^gradients/Kelz/fc5/Tensordot/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/Kelz/fc5/Tensordot/MatMul_grad/MatMul*(
_output_shapes
:         ђ,
»
Cgradients/Kelz/fc5/Tensordot/MatMul_grad/tuple/control_dependency_1Identity1gradients/Kelz/fc5/Tensordot/MatMul_grad/MatMul_1:^gradients/Kelz/fc5/Tensordot/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Kelz/fc5/Tensordot/MatMul_grad/MatMul_1* 
_output_shapes
:
ђ,ђ
І
/gradients/Kelz/fc5/Tensordot/Reshape_grad/ShapeShapeKelz/fc5/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
є
1gradients/Kelz/fc5/Tensordot/Reshape_grad/ReshapeReshapeAgradients/Kelz/fc5/Tensordot/MatMul_grad/tuple/control_dependency/gradients/Kelz/fc5/Tensordot/Reshape_grad/Shape*=
_output_shapes+
):'                           *
T0*
Tshape0
ѓ
1gradients/Kelz/fc5/Tensordot/Reshape_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
№
3gradients/Kelz/fc5/Tensordot/Reshape_1_grad/ReshapeReshapeCgradients/Kelz/fc5/Tensordot/MatMul_grad/tuple/control_dependency_11gradients/Kelz/fc5/Tensordot/Reshape_1_grad/Shape* 
_output_shapes
:
ђ,ђ*
T0*
Tshape0
Ю
=gradients/Kelz/fc5/Tensordot/transpose_grad/InvertPermutationInvertPermutationKelz/fc5/Tensordot/concat_1*#
_output_shapes
:         *
T0
Ђ
5gradients/Kelz/fc5/Tensordot/transpose_grad/transpose	Transpose1gradients/Kelz/fc5/Tensordot/Reshape_grad/Reshape=gradients/Kelz/fc5/Tensordot/transpose_grad/InvertPermutation*
T0*5
_output_shapes#
!:                  ђ,*
Tperm0
ъ
?gradients/Kelz/fc5/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation#Kelz/fc5/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
Ы
7gradients/Kelz/fc5/Tensordot/transpose_1_grad/transpose	Transpose3gradients/Kelz/fc5/Tensordot/Reshape_1_grad/Reshape?gradients/Kelz/fc5/Tensordot/transpose_1_grad/InvertPermutation*
T0* 
_output_shapes
:
ђ,ђ*
Tperm0
{
"gradients/Kelz/flatten4_grad/ShapeShapeKelz/dropout3/dropout/mul*
T0*
out_type0*
_output_shapes
:
█
$gradients/Kelz/flatten4_grad/ReshapeReshape5gradients/Kelz/fc5/Tensordot/transpose_grad/transpose"gradients/Kelz/flatten4_grad/Shape*
T0*
Tshape0*8
_output_shapes&
$:"                  X@
Є
.gradients/Kelz/dropout3/dropout/mul_grad/ShapeShapeKelz/dropout3/dropout/div*
_output_shapes
:*
T0*
out_type0
І
0gradients/Kelz/dropout3/dropout/mul_grad/Shape_1ShapeKelz/dropout3/dropout/Floor*
T0*
out_type0*
_output_shapes
:
Ш
>gradients/Kelz/dropout3/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/Kelz/dropout3/dropout/mul_grad/Shape0gradients/Kelz/dropout3/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╣
,gradients/Kelz/dropout3/dropout/mul_grad/mulMul$gradients/Kelz/flatten4_grad/ReshapeKelz/dropout3/dropout/Floor*
T0*8
_output_shapes&
$:"                  X@
р
,gradients/Kelz/dropout3/dropout/mul_grad/SumSum,gradients/Kelz/dropout3/dropout/mul_grad/mul>gradients/Kelz/dropout3/dropout/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ж
0gradients/Kelz/dropout3/dropout/mul_grad/ReshapeReshape,gradients/Kelz/dropout3/dropout/mul_grad/Sum.gradients/Kelz/dropout3/dropout/mul_grad/Shape*
T0*
Tshape0*8
_output_shapes&
$:"                  X@
╣
.gradients/Kelz/dropout3/dropout/mul_grad/mul_1MulKelz/dropout3/dropout/div$gradients/Kelz/flatten4_grad/Reshape*
T0*8
_output_shapes&
$:"                  X@
у
.gradients/Kelz/dropout3/dropout/mul_grad/Sum_1Sum.gradients/Kelz/dropout3/dropout/mul_grad/mul_1@gradients/Kelz/dropout3/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
­
2gradients/Kelz/dropout3/dropout/mul_grad/Reshape_1Reshape.gradients/Kelz/dropout3/dropout/mul_grad/Sum_10gradients/Kelz/dropout3/dropout/mul_grad/Shape_1*
T0*
Tshape0*8
_output_shapes&
$:"                  X@
Е
9gradients/Kelz/dropout3/dropout/mul_grad/tuple/group_depsNoOp1^gradients/Kelz/dropout3/dropout/mul_grad/Reshape3^gradients/Kelz/dropout3/dropout/mul_grad/Reshape_1
├
Agradients/Kelz/dropout3/dropout/mul_grad/tuple/control_dependencyIdentity0gradients/Kelz/dropout3/dropout/mul_grad/Reshape:^gradients/Kelz/dropout3/dropout/mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Kelz/dropout3/dropout/mul_grad/Reshape*8
_output_shapes&
$:"                  X@
╔
Cgradients/Kelz/dropout3/dropout/mul_grad/tuple/control_dependency_1Identity2gradients/Kelz/dropout3/dropout/mul_grad/Reshape_1:^gradients/Kelz/dropout3/dropout/mul_grad/tuple/group_deps*8
_output_shapes&
$:"                  X@*
T0*E
_class;
97loc:@gradients/Kelz/dropout3/dropout/mul_grad/Reshape_1
ђ
.gradients/Kelz/dropout3/dropout/div_grad/ShapeShapeKelz/pool3/MaxPool*
T0*
out_type0*
_output_shapes
:
s
0gradients/Kelz/dropout3/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ш
>gradients/Kelz/dropout3/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/Kelz/dropout3/dropout/div_grad/Shape0gradients/Kelz/dropout3/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Р
0gradients/Kelz/dropout3/dropout/div_grad/RealDivRealDivAgradients/Kelz/dropout3/dropout/mul_grad/tuple/control_dependencyKelz/dropout3/dropout/keep_prob*
T0*8
_output_shapes&
$:"                  X@
т
,gradients/Kelz/dropout3/dropout/div_grad/SumSum0gradients/Kelz/dropout3/dropout/div_grad/RealDiv>gradients/Kelz/dropout3/dropout/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ж
0gradients/Kelz/dropout3/dropout/div_grad/ReshapeReshape,gradients/Kelz/dropout3/dropout/div_grad/Sum.gradients/Kelz/dropout3/dropout/div_grad/Shape*
T0*
Tshape0*8
_output_shapes&
$:"                  X@
і
,gradients/Kelz/dropout3/dropout/div_grad/NegNegKelz/pool3/MaxPool*
T0*8
_output_shapes&
$:"                  X@
¤
2gradients/Kelz/dropout3/dropout/div_grad/RealDiv_1RealDiv,gradients/Kelz/dropout3/dropout/div_grad/NegKelz/dropout3/dropout/keep_prob*
T0*8
_output_shapes&
$:"                  X@
Н
2gradients/Kelz/dropout3/dropout/div_grad/RealDiv_2RealDiv2gradients/Kelz/dropout3/dropout/div_grad/RealDiv_1Kelz/dropout3/dropout/keep_prob*
T0*8
_output_shapes&
$:"                  X@
ь
,gradients/Kelz/dropout3/dropout/div_grad/mulMulAgradients/Kelz/dropout3/dropout/mul_grad/tuple/control_dependency2gradients/Kelz/dropout3/dropout/div_grad/RealDiv_2*
T0*8
_output_shapes&
$:"                  X@
т
.gradients/Kelz/dropout3/dropout/div_grad/Sum_1Sum,gradients/Kelz/dropout3/dropout/div_grad/mul@gradients/Kelz/dropout3/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
╬
2gradients/Kelz/dropout3/dropout/div_grad/Reshape_1Reshape.gradients/Kelz/dropout3/dropout/div_grad/Sum_10gradients/Kelz/dropout3/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Е
9gradients/Kelz/dropout3/dropout/div_grad/tuple/group_depsNoOp1^gradients/Kelz/dropout3/dropout/div_grad/Reshape3^gradients/Kelz/dropout3/dropout/div_grad/Reshape_1
├
Agradients/Kelz/dropout3/dropout/div_grad/tuple/control_dependencyIdentity0gradients/Kelz/dropout3/dropout/div_grad/Reshape:^gradients/Kelz/dropout3/dropout/div_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Kelz/dropout3/dropout/div_grad/Reshape*8
_output_shapes&
$:"                  X@
Д
Cgradients/Kelz/dropout3/dropout/div_grad/tuple/control_dependency_1Identity2gradients/Kelz/dropout3/dropout/div_grad/Reshape_1:^gradients/Kelz/dropout3/dropout/div_grad/tuple/group_deps*
_output_shapes
: *
T0*E
_class;
97loc:@gradients/Kelz/dropout3/dropout/div_grad/Reshape_1
╗
-gradients/Kelz/pool3/MaxPool_grad/MaxPoolGradMaxPoolGradKelz/conv3/ReluKelz/pool3/MaxPoolAgradients/Kelz/dropout3/dropout/div_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*9
_output_shapes'
%:#                  ░@
и
'gradients/Kelz/conv3/Relu_grad/ReluGradReluGrad-gradients/Kelz/pool3/MaxPool_grad/MaxPoolGradKelz/conv3/Relu*
T0*9
_output_shapes'
%:#                  ░@
А
-gradients/Kelz/conv3/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/Kelz/conv3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
ћ
2gradients/Kelz/conv3/BiasAdd_grad/tuple/group_depsNoOp(^gradients/Kelz/conv3/Relu_grad/ReluGrad.^gradients/Kelz/conv3/BiasAdd_grad/BiasAddGrad
ц
:gradients/Kelz/conv3/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/Kelz/conv3/Relu_grad/ReluGrad3^gradients/Kelz/conv3/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/Kelz/conv3/Relu_grad/ReluGrad*9
_output_shapes'
%:#                  ░@
Њ
<gradients/Kelz/conv3/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/Kelz/conv3/BiasAdd_grad/BiasAddGrad3^gradients/Kelz/conv3/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Kelz/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
ц
'gradients/Kelz/conv3/Conv2D_grad/ShapeNShapeNKelz/dropout2/dropout/mulconv3/weights/read*
N* 
_output_shapes
::*
T0*
out_type0

&gradients/Kelz/conv3/Conv2D_grad/ConstConst*%
valueB"          @   *
dtype0*
_output_shapes
:
Є
4gradients/Kelz/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/Kelz/conv3/Conv2D_grad/ShapeNconv3/weights/read:gradients/Kelz/conv3/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
в
5gradients/Kelz/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterKelz/dropout2/dropout/mul&gradients/Kelz/conv3/Conv2D_grad/Const:gradients/Kelz/conv3/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @
е
1gradients/Kelz/conv3/Conv2D_grad/tuple/group_depsNoOp5^gradients/Kelz/conv3/Conv2D_grad/Conv2DBackpropInput6^gradients/Kelz/conv3/Conv2D_grad/Conv2DBackpropFilter
╝
9gradients/Kelz/conv3/Conv2D_grad/tuple/control_dependencyIdentity4gradients/Kelz/conv3/Conv2D_grad/Conv2DBackpropInput2^gradients/Kelz/conv3/Conv2D_grad/tuple/group_deps*9
_output_shapes'
%:#                  ░ *
T0*G
_class=
;9loc:@gradients/Kelz/conv3/Conv2D_grad/Conv2DBackpropInput
Г
;gradients/Kelz/conv3/Conv2D_grad/tuple/control_dependency_1Identity5gradients/Kelz/conv3/Conv2D_grad/Conv2DBackpropFilter2^gradients/Kelz/conv3/Conv2D_grad/tuple/group_deps*&
_output_shapes
: @*
T0*H
_class>
<:loc:@gradients/Kelz/conv3/Conv2D_grad/Conv2DBackpropFilter
Є
.gradients/Kelz/dropout2/dropout/mul_grad/ShapeShapeKelz/dropout2/dropout/div*
T0*
out_type0*
_output_shapes
:
І
0gradients/Kelz/dropout2/dropout/mul_grad/Shape_1ShapeKelz/dropout2/dropout/Floor*
T0*
out_type0*
_output_shapes
:
Ш
>gradients/Kelz/dropout2/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/Kelz/dropout2/dropout/mul_grad/Shape0gradients/Kelz/dropout2/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
¤
,gradients/Kelz/dropout2/dropout/mul_grad/mulMul9gradients/Kelz/conv3/Conv2D_grad/tuple/control_dependencyKelz/dropout2/dropout/Floor*
T0*9
_output_shapes'
%:#                  ░ 
р
,gradients/Kelz/dropout2/dropout/mul_grad/SumSum,gradients/Kelz/dropout2/dropout/mul_grad/mul>gradients/Kelz/dropout2/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
в
0gradients/Kelz/dropout2/dropout/mul_grad/ReshapeReshape,gradients/Kelz/dropout2/dropout/mul_grad/Sum.gradients/Kelz/dropout2/dropout/mul_grad/Shape*9
_output_shapes'
%:#                  ░ *
T0*
Tshape0
¤
.gradients/Kelz/dropout2/dropout/mul_grad/mul_1MulKelz/dropout2/dropout/div9gradients/Kelz/conv3/Conv2D_grad/tuple/control_dependency*
T0*9
_output_shapes'
%:#                  ░ 
у
.gradients/Kelz/dropout2/dropout/mul_grad/Sum_1Sum.gradients/Kelz/dropout2/dropout/mul_grad/mul_1@gradients/Kelz/dropout2/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
2gradients/Kelz/dropout2/dropout/mul_grad/Reshape_1Reshape.gradients/Kelz/dropout2/dropout/mul_grad/Sum_10gradients/Kelz/dropout2/dropout/mul_grad/Shape_1*9
_output_shapes'
%:#                  ░ *
T0*
Tshape0
Е
9gradients/Kelz/dropout2/dropout/mul_grad/tuple/group_depsNoOp1^gradients/Kelz/dropout2/dropout/mul_grad/Reshape3^gradients/Kelz/dropout2/dropout/mul_grad/Reshape_1
─
Agradients/Kelz/dropout2/dropout/mul_grad/tuple/control_dependencyIdentity0gradients/Kelz/dropout2/dropout/mul_grad/Reshape:^gradients/Kelz/dropout2/dropout/mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Kelz/dropout2/dropout/mul_grad/Reshape*9
_output_shapes'
%:#                  ░ 
╩
Cgradients/Kelz/dropout2/dropout/mul_grad/tuple/control_dependency_1Identity2gradients/Kelz/dropout2/dropout/mul_grad/Reshape_1:^gradients/Kelz/dropout2/dropout/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/Kelz/dropout2/dropout/mul_grad/Reshape_1*9
_output_shapes'
%:#                  ░ 
ђ
.gradients/Kelz/dropout2/dropout/div_grad/ShapeShapeKelz/pool2/MaxPool*
T0*
out_type0*
_output_shapes
:
s
0gradients/Kelz/dropout2/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ш
>gradients/Kelz/dropout2/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/Kelz/dropout2/dropout/div_grad/Shape0gradients/Kelz/dropout2/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
с
0gradients/Kelz/dropout2/dropout/div_grad/RealDivRealDivAgradients/Kelz/dropout2/dropout/mul_grad/tuple/control_dependencyKelz/dropout2/dropout/keep_prob*
T0*9
_output_shapes'
%:#                  ░ 
т
,gradients/Kelz/dropout2/dropout/div_grad/SumSum0gradients/Kelz/dropout2/dropout/div_grad/RealDiv>gradients/Kelz/dropout2/dropout/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
в
0gradients/Kelz/dropout2/dropout/div_grad/ReshapeReshape,gradients/Kelz/dropout2/dropout/div_grad/Sum.gradients/Kelz/dropout2/dropout/div_grad/Shape*
T0*
Tshape0*9
_output_shapes'
%:#                  ░ 
І
,gradients/Kelz/dropout2/dropout/div_grad/NegNegKelz/pool2/MaxPool*
T0*9
_output_shapes'
%:#                  ░ 
л
2gradients/Kelz/dropout2/dropout/div_grad/RealDiv_1RealDiv,gradients/Kelz/dropout2/dropout/div_grad/NegKelz/dropout2/dropout/keep_prob*9
_output_shapes'
%:#                  ░ *
T0
о
2gradients/Kelz/dropout2/dropout/div_grad/RealDiv_2RealDiv2gradients/Kelz/dropout2/dropout/div_grad/RealDiv_1Kelz/dropout2/dropout/keep_prob*
T0*9
_output_shapes'
%:#                  ░ 
Ь
,gradients/Kelz/dropout2/dropout/div_grad/mulMulAgradients/Kelz/dropout2/dropout/mul_grad/tuple/control_dependency2gradients/Kelz/dropout2/dropout/div_grad/RealDiv_2*
T0*9
_output_shapes'
%:#                  ░ 
т
.gradients/Kelz/dropout2/dropout/div_grad/Sum_1Sum,gradients/Kelz/dropout2/dropout/div_grad/mul@gradients/Kelz/dropout2/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
╬
2gradients/Kelz/dropout2/dropout/div_grad/Reshape_1Reshape.gradients/Kelz/dropout2/dropout/div_grad/Sum_10gradients/Kelz/dropout2/dropout/div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
Е
9gradients/Kelz/dropout2/dropout/div_grad/tuple/group_depsNoOp1^gradients/Kelz/dropout2/dropout/div_grad/Reshape3^gradients/Kelz/dropout2/dropout/div_grad/Reshape_1
─
Agradients/Kelz/dropout2/dropout/div_grad/tuple/control_dependencyIdentity0gradients/Kelz/dropout2/dropout/div_grad/Reshape:^gradients/Kelz/dropout2/dropout/div_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Kelz/dropout2/dropout/div_grad/Reshape*9
_output_shapes'
%:#                  ░ 
Д
Cgradients/Kelz/dropout2/dropout/div_grad/tuple/control_dependency_1Identity2gradients/Kelz/dropout2/dropout/div_grad/Reshape_1:^gradients/Kelz/dropout2/dropout/div_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/Kelz/dropout2/dropout/div_grad/Reshape_1*
_output_shapes
: 
╗
-gradients/Kelz/pool2/MaxPool_grad/MaxPoolGradMaxPoolGradKelz/conv2/ReluKelz/pool2/MaxPoolAgradients/Kelz/dropout2/dropout/div_grad/tuple/control_dependency*
ksize
*
paddingVALID*9
_output_shapes'
%:#                  Я *
T0*
data_formatNHWC*
strides

и
'gradients/Kelz/conv2/Relu_grad/ReluGradReluGrad-gradients/Kelz/pool2/MaxPool_grad/MaxPoolGradKelz/conv2/Relu*
T0*9
_output_shapes'
%:#                  Я 
m
gradients/zeros_like	ZerosLike%Kelz/conv2/BatchNorm/FusedBatchNorm:1*
_output_shapes
: *
T0
o
gradients/zeros_like_1	ZerosLike%Kelz/conv2/BatchNorm/FusedBatchNorm:2*
_output_shapes
: *
T0
o
gradients/zeros_like_2	ZerosLike%Kelz/conv2/BatchNorm/FusedBatchNorm:3*
T0*
_output_shapes
: 
o
gradients/zeros_like_3	ZerosLike%Kelz/conv2/BatchNorm/FusedBatchNorm:4*
T0*
_output_shapes
: 
Ћ
Egradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad'gradients/Kelz/conv2/Relu_grad/ReluGradKelz/conv2/Conv2DKelz/conv2/BatchNorm/Const%Kelz/conv2/BatchNorm/FusedBatchNorm:3%Kelz/conv2/BatchNorm/FusedBatchNorm:4*
epsilon%oЃ:*
T0*
data_formatNHWC*M
_output_shapes;
9:#                  Я : : : : *
is_training(
Њ
Cgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/group_depsNoOpF^gradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad
ѓ
Kgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/control_dependencyIdentityEgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGradD^gradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad*9
_output_shapes'
%:#                  Я 
у
Mgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency_1IdentityGgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad:1D^gradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
у
Mgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency_2IdentityGgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad:2D^gradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
т
Mgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency_3IdentityGgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad:3D^gradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
т
Mgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency_4IdentityGgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad:4D^gradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
џ
'gradients/Kelz/conv2/Conv2D_grad/ShapeNShapeNKelz/conv1/Reluconv2/weights/read*
T0*
out_type0*
N* 
_output_shapes
::

&gradients/Kelz/conv2/Conv2D_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"              
ў
4gradients/Kelz/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/Kelz/conv2/Conv2D_grad/ShapeNconv2/weights/readKgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4                                    
Ы
5gradients/Kelz/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterKelz/conv1/Relu&gradients/Kelz/conv2/Conv2D_grad/ConstKgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:  
е
1gradients/Kelz/conv2/Conv2D_grad/tuple/group_depsNoOp5^gradients/Kelz/conv2/Conv2D_grad/Conv2DBackpropInput6^gradients/Kelz/conv2/Conv2D_grad/Conv2DBackpropFilter
╝
9gradients/Kelz/conv2/Conv2D_grad/tuple/control_dependencyIdentity4gradients/Kelz/conv2/Conv2D_grad/Conv2DBackpropInput2^gradients/Kelz/conv2/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/Kelz/conv2/Conv2D_grad/Conv2DBackpropInput*9
_output_shapes'
%:#                  Я 
Г
;gradients/Kelz/conv2/Conv2D_grad/tuple/control_dependency_1Identity5gradients/Kelz/conv2/Conv2D_grad/Conv2DBackpropFilter2^gradients/Kelz/conv2/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Kelz/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:  
├
'gradients/Kelz/conv1/Relu_grad/ReluGradReluGrad9gradients/Kelz/conv2/Conv2D_grad/tuple/control_dependencyKelz/conv1/Relu*
T0*9
_output_shapes'
%:#                  Я 
А
-gradients/Kelz/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/Kelz/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
ћ
2gradients/Kelz/conv1/BiasAdd_grad/tuple/group_depsNoOp(^gradients/Kelz/conv1/Relu_grad/ReluGrad.^gradients/Kelz/conv1/BiasAdd_grad/BiasAddGrad
ц
:gradients/Kelz/conv1/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/Kelz/conv1/Relu_grad/ReluGrad3^gradients/Kelz/conv1/BiasAdd_grad/tuple/group_deps*9
_output_shapes'
%:#                  Я *
T0*:
_class0
.,loc:@gradients/Kelz/conv1/Relu_grad/ReluGrad
Њ
<gradients/Kelz/conv1/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/Kelz/conv1/BiasAdd_grad/BiasAddGrad3^gradients/Kelz/conv1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Kelz/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
љ
'gradients/Kelz/conv1/Conv2D_grad/ShapeNShapeNinputconv1/weights/read*
T0*
out_type0*
N* 
_output_shapes
::

&gradients/Kelz/conv1/Conv2D_grad/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
Є
4gradients/Kelz/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/Kelz/conv1/Conv2D_grad/ShapeNconv1/weights/read:gradients/Kelz/conv1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*J
_output_shapes8
6:4                                    *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
О
5gradients/Kelz/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput&gradients/Kelz/conv1/Conv2D_grad/Const:gradients/Kelz/conv1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
е
1gradients/Kelz/conv1/Conv2D_grad/tuple/group_depsNoOp5^gradients/Kelz/conv1/Conv2D_grad/Conv2DBackpropInput6^gradients/Kelz/conv1/Conv2D_grad/Conv2DBackpropFilter
╝
9gradients/Kelz/conv1/Conv2D_grad/tuple/control_dependencyIdentity4gradients/Kelz/conv1/Conv2D_grad/Conv2DBackpropInput2^gradients/Kelz/conv1/Conv2D_grad/tuple/group_deps*9
_output_shapes'
%:#                  Я*
T0*G
_class=
;9loc:@gradients/Kelz/conv1/Conv2D_grad/Conv2DBackpropInput
Г
;gradients/Kelz/conv1/Conv2D_grad/tuple/control_dependency_1Identity5gradients/Kelz/conv1/Conv2D_grad/Conv2DBackpropFilter2^gradients/Kelz/conv1/Conv2D_grad/tuple/group_deps*&
_output_shapes
: *
T0*H
_class>
<:loc:@gradients/Kelz/conv1/Conv2D_grad/Conv2DBackpropFilter

beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@conv1/biases
љ
beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@conv1/biases*
	container *
shape: 
»
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: *
use_locking(
k
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 

beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wЙ?*
_class
loc:@conv1/biases
љ
beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@conv1/biases*
	container *
shape: 
»
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
k
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
»
4conv1/weights/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"             * 
_class
loc:@conv1/weights
Љ
*conv1/weights/Adam/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 
з
$conv1/weights/Adam/Initializer/zerosFill4conv1/weights/Adam/Initializer/zeros/shape_as_tensor*conv1/weights/Adam/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
И
conv1/weights/Adam
VariableV2*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name * 
_class
loc:@conv1/weights
┘
conv1/weights/Adam/AssignAssignconv1/weights/Adam$conv1/weights/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
і
conv1/weights/Adam/readIdentityconv1/weights/Adam*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
▒
6conv1/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"             * 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
:
Њ
,conv1/weights/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 
щ
&conv1/weights/Adam_1/Initializer/zerosFill6conv1/weights/Adam_1/Initializer/zeros/shape_as_tensor,conv1/weights/Adam_1/Initializer/zeros/Const*&
_output_shapes
: *
T0*

index_type0* 
_class
loc:@conv1/weights
║
conv1/weights/Adam_1
VariableV2*
shape: *
dtype0*&
_output_shapes
: *
shared_name * 
_class
loc:@conv1/weights*
	container 
▀
conv1/weights/Adam_1/AssignAssignconv1/weights/Adam_1&conv1/weights/Adam_1/Initializer/zeros*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: *
use_locking(
ј
conv1/weights/Adam_1/readIdentityconv1/weights/Adam_1*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
: 
ъ
3conv1/biases/Adam/Initializer/zeros/shape_as_tensorConst*
valueB: *
_class
loc:@conv1/biases*
dtype0*
_output_shapes
:
Ј
)conv1/biases/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@conv1/biases*
dtype0*
_output_shapes
: 
с
#conv1/biases/Adam/Initializer/zerosFill3conv1/biases/Adam/Initializer/zeros/shape_as_tensor)conv1/biases/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@conv1/biases*
_output_shapes
: 
ъ
conv1/biases/Adam
VariableV2*
_class
loc:@conv1/biases*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
╔
conv1/biases/Adam/AssignAssignconv1/biases/Adam#conv1/biases/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
{
conv1/biases/Adam/readIdentityconv1/biases/Adam*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
а
5conv1/biases/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB: *
_class
loc:@conv1/biases*
dtype0*
_output_shapes
:
Љ
+conv1/biases/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@conv1/biases*
dtype0*
_output_shapes
: 
ж
%conv1/biases/Adam_1/Initializer/zerosFill5conv1/biases/Adam_1/Initializer/zeros/shape_as_tensor+conv1/biases/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@conv1/biases*
_output_shapes
: 
а
conv1/biases/Adam_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@conv1/biases*
	container *
shape: 
¤
conv1/biases/Adam_1/AssignAssignconv1/biases/Adam_1%conv1/biases/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases

conv1/biases/Adam_1/readIdentityconv1/biases/Adam_1*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
»
4conv2/weights/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"              * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
:
Љ
*conv2/weights/Adam/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
: 
з
$conv2/weights/Adam/Initializer/zerosFill4conv2/weights/Adam/Initializer/zeros/shape_as_tensor*conv2/weights/Adam/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv2/weights*&
_output_shapes
:  
И
conv2/weights/Adam
VariableV2*
	container *
shape:  *
dtype0*&
_output_shapes
:  *
shared_name * 
_class
loc:@conv2/weights
┘
conv2/weights/Adam/AssignAssignconv2/weights/Adam$conv2/weights/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
:  
і
conv2/weights/Adam/readIdentityconv2/weights/Adam*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
:  
▒
6conv2/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"              * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
:
Њ
,conv2/weights/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    * 
_class
loc:@conv2/weights
щ
&conv2/weights/Adam_1/Initializer/zerosFill6conv2/weights/Adam_1/Initializer/zeros/shape_as_tensor,conv2/weights/Adam_1/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv2/weights*&
_output_shapes
:  
║
conv2/weights/Adam_1
VariableV2*
shared_name * 
_class
loc:@conv2/weights*
	container *
shape:  *
dtype0*&
_output_shapes
:  
▀
conv2/weights/Adam_1/AssignAssignconv2/weights/Adam_1&conv2/weights/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
:  
ј
conv2/weights/Adam_1/readIdentityconv2/weights/Adam_1*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
:  
«
;conv2/BatchNorm/beta/Adam/Initializer/zeros/shape_as_tensorConst*
valueB: *'
_class
loc:@conv2/BatchNorm/beta*
dtype0*
_output_shapes
:
Ъ
1conv2/BatchNorm/beta/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *'
_class
loc:@conv2/BatchNorm/beta
Ѓ
+conv2/BatchNorm/beta/Adam/Initializer/zerosFill;conv2/BatchNorm/beta/Adam/Initializer/zeros/shape_as_tensor1conv2/BatchNorm/beta/Adam/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@conv2/BatchNorm/beta*
_output_shapes
: 
«
conv2/BatchNorm/beta/Adam
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@conv2/BatchNorm/beta*
	container 
ж
 conv2/BatchNorm/beta/Adam/AssignAssignconv2/BatchNorm/beta/Adam+conv2/BatchNorm/beta/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@conv2/BatchNorm/beta*
validate_shape(*
_output_shapes
: 
Њ
conv2/BatchNorm/beta/Adam/readIdentityconv2/BatchNorm/beta/Adam*
T0*'
_class
loc:@conv2/BatchNorm/beta*
_output_shapes
: 
░
=conv2/BatchNorm/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB: *'
_class
loc:@conv2/BatchNorm/beta*
dtype0*
_output_shapes
:
А
3conv2/BatchNorm/beta/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@conv2/BatchNorm/beta*
dtype0*
_output_shapes
: 
Ѕ
-conv2/BatchNorm/beta/Adam_1/Initializer/zerosFill=conv2/BatchNorm/beta/Adam_1/Initializer/zeros/shape_as_tensor3conv2/BatchNorm/beta/Adam_1/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@conv2/BatchNorm/beta*
_output_shapes
: 
░
conv2/BatchNorm/beta/Adam_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@conv2/BatchNorm/beta*
	container 
№
"conv2/BatchNorm/beta/Adam_1/AssignAssignconv2/BatchNorm/beta/Adam_1-conv2/BatchNorm/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@conv2/BatchNorm/beta*
validate_shape(*
_output_shapes
: 
Ќ
 conv2/BatchNorm/beta/Adam_1/readIdentityconv2/BatchNorm/beta/Adam_1*
T0*'
_class
loc:@conv2/BatchNorm/beta*
_output_shapes
: 
»
4conv3/weights/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   * 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
:
Љ
*conv3/weights/Adam/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
: 
з
$conv3/weights/Adam/Initializer/zerosFill4conv3/weights/Adam/Initializer/zeros/shape_as_tensor*conv3/weights/Adam/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv3/weights*&
_output_shapes
: @
И
conv3/weights/Adam
VariableV2* 
_class
loc:@conv3/weights*
	container *
shape: @*
dtype0*&
_output_shapes
: @*
shared_name 
┘
conv3/weights/Adam/AssignAssignconv3/weights/Adam$conv3/weights/Adam/Initializer/zeros*
T0* 
_class
loc:@conv3/weights*
validate_shape(*&
_output_shapes
: @*
use_locking(
і
conv3/weights/Adam/readIdentityconv3/weights/Adam*
T0* 
_class
loc:@conv3/weights*&
_output_shapes
: @
▒
6conv3/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   * 
_class
loc:@conv3/weights*
dtype0*
_output_shapes
:
Њ
,conv3/weights/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    * 
_class
loc:@conv3/weights
щ
&conv3/weights/Adam_1/Initializer/zerosFill6conv3/weights/Adam_1/Initializer/zeros/shape_as_tensor,conv3/weights/Adam_1/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@conv3/weights*&
_output_shapes
: @
║
conv3/weights/Adam_1
VariableV2*
	container *
shape: @*
dtype0*&
_output_shapes
: @*
shared_name * 
_class
loc:@conv3/weights
▀
conv3/weights/Adam_1/AssignAssignconv3/weights/Adam_1&conv3/weights/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*&
_output_shapes
: @
ј
conv3/weights/Adam_1/readIdentityconv3/weights/Adam_1*
T0* 
_class
loc:@conv3/weights*&
_output_shapes
: @
ъ
3conv3/biases/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:@*
_class
loc:@conv3/biases*
dtype0*
_output_shapes
:
Ј
)conv3/biases/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@conv3/biases*
dtype0*
_output_shapes
: 
с
#conv3/biases/Adam/Initializer/zerosFill3conv3/biases/Adam/Initializer/zeros/shape_as_tensor)conv3/biases/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@conv3/biases*
_output_shapes
:@
ъ
conv3/biases/Adam
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv3/biases
╔
conv3/biases/Adam/AssignAssignconv3/biases/Adam#conv3/biases/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv3/biases
{
conv3/biases/Adam/readIdentityconv3/biases/Adam*
T0*
_class
loc:@conv3/biases*
_output_shapes
:@
а
5conv3/biases/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:@*
_class
loc:@conv3/biases
Љ
+conv3/biases/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@conv3/biases
ж
%conv3/biases/Adam_1/Initializer/zerosFill5conv3/biases/Adam_1/Initializer/zeros/shape_as_tensor+conv3/biases/Adam_1/Initializer/zeros/Const*
_output_shapes
:@*
T0*

index_type0*
_class
loc:@conv3/biases
а
conv3/biases/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv3/biases*
	container 
¤
conv3/biases/Adam_1/AssignAssignconv3/biases/Adam_1%conv3/biases/Adam_1/Initializer/zeros*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes
:@*
use_locking(

conv3/biases/Adam_1/readIdentityconv3/biases/Adam_1*
T0*
_class
loc:@conv3/biases*
_output_shapes
:@
Б
2fc5/weights/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *
_class
loc:@fc5/weights*
dtype0*
_output_shapes
:
Ї
(fc5/weights/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@fc5/weights
т
"fc5/weights/Adam/Initializer/zerosFill2fc5/weights/Adam/Initializer/zeros/shape_as_tensor(fc5/weights/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@fc5/weights* 
_output_shapes
:
ђ,ђ
е
fc5/weights/Adam
VariableV2*
dtype0* 
_output_shapes
:
ђ,ђ*
shared_name *
_class
loc:@fc5/weights*
	container *
shape:
ђ,ђ
╦
fc5/weights/Adam/AssignAssignfc5/weights/Adam"fc5/weights/Adam/Initializer/zeros*
T0*
_class
loc:@fc5/weights*
validate_shape(* 
_output_shapes
:
ђ,ђ*
use_locking(
~
fc5/weights/Adam/readIdentityfc5/weights/Adam* 
_output_shapes
:
ђ,ђ*
T0*
_class
loc:@fc5/weights
Ц
4fc5/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"      *
_class
loc:@fc5/weights
Ј
*fc5/weights/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@fc5/weights*
dtype0*
_output_shapes
: 
в
$fc5/weights/Adam_1/Initializer/zerosFill4fc5/weights/Adam_1/Initializer/zeros/shape_as_tensor*fc5/weights/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@fc5/weights* 
_output_shapes
:
ђ,ђ
ф
fc5/weights/Adam_1
VariableV2*
_class
loc:@fc5/weights*
	container *
shape:
ђ,ђ*
dtype0* 
_output_shapes
:
ђ,ђ*
shared_name 
Л
fc5/weights/Adam_1/AssignAssignfc5/weights/Adam_1$fc5/weights/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@fc5/weights*
validate_shape(* 
_output_shapes
:
ђ,ђ
ѓ
fc5/weights/Adam_1/readIdentityfc5/weights/Adam_1* 
_output_shapes
:
ђ,ђ*
T0*
_class
loc:@fc5/weights
Џ
1fc5/biases/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:ђ*
_class
loc:@fc5/biases*
dtype0*
_output_shapes
:
І
'fc5/biases/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@fc5/biases*
dtype0*
_output_shapes
: 
▄
!fc5/biases/Adam/Initializer/zerosFill1fc5/biases/Adam/Initializer/zeros/shape_as_tensor'fc5/biases/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@fc5/biases*
_output_shapes	
:ђ
ю
fc5/biases/Adam
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *
_class
loc:@fc5/biases
┬
fc5/biases/Adam/AssignAssignfc5/biases/Adam!fc5/biases/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*
_class
loc:@fc5/biases
v
fc5/biases/Adam/readIdentityfc5/biases/Adam*
_output_shapes	
:ђ*
T0*
_class
loc:@fc5/biases
Ю
3fc5/biases/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:ђ*
_class
loc:@fc5/biases
Ї
)fc5/biases/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@fc5/biases*
dtype0*
_output_shapes
: 
Р
#fc5/biases/Adam_1/Initializer/zerosFill3fc5/biases/Adam_1/Initializer/zeros/shape_as_tensor)fc5/biases/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@fc5/biases*
_output_shapes	
:ђ
ъ
fc5/biases/Adam_1
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *
_class
loc:@fc5/biases*
	container 
╚
fc5/biases/Adam_1/AssignAssignfc5/biases/Adam_1#fc5/biases/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@fc5/biases*
validate_shape(*
_output_shapes	
:ђ
z
fc5/biases/Adam_1/readIdentityfc5/biases/Adam_1*
T0*
_class
loc:@fc5/biases*
_output_shapes	
:ђ
Б
2fc6/weights/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"   X   *
_class
loc:@fc6/weights*
dtype0*
_output_shapes
:
Ї
(fc6/weights/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@fc6/weights
С
"fc6/weights/Adam/Initializer/zerosFill2fc6/weights/Adam/Initializer/zeros/shape_as_tensor(fc6/weights/Adam/Initializer/zeros/Const*
_output_shapes
:	ђX*
T0*

index_type0*
_class
loc:@fc6/weights
д
fc6/weights/Adam
VariableV2*
shape:	ђX*
dtype0*
_output_shapes
:	ђX*
shared_name *
_class
loc:@fc6/weights*
	container 
╩
fc6/weights/Adam/AssignAssignfc6/weights/Adam"fc6/weights/Adam/Initializer/zeros*
T0*
_class
loc:@fc6/weights*
validate_shape(*
_output_shapes
:	ђX*
use_locking(
}
fc6/weights/Adam/readIdentityfc6/weights/Adam*
T0*
_class
loc:@fc6/weights*
_output_shapes
:	ђX
Ц
4fc6/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   X   *
_class
loc:@fc6/weights*
dtype0*
_output_shapes
:
Ј
*fc6/weights/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@fc6/weights
Ж
$fc6/weights/Adam_1/Initializer/zerosFill4fc6/weights/Adam_1/Initializer/zeros/shape_as_tensor*fc6/weights/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@fc6/weights*
_output_shapes
:	ђX
е
fc6/weights/Adam_1
VariableV2*
shared_name *
_class
loc:@fc6/weights*
	container *
shape:	ђX*
dtype0*
_output_shapes
:	ђX
л
fc6/weights/Adam_1/AssignAssignfc6/weights/Adam_1$fc6/weights/Adam_1/Initializer/zeros*
T0*
_class
loc:@fc6/weights*
validate_shape(*
_output_shapes
:	ђX*
use_locking(
Ђ
fc6/weights/Adam_1/readIdentityfc6/weights/Adam_1*
_output_shapes
:	ђX*
T0*
_class
loc:@fc6/weights
џ
1fc6/biases/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:X*
_class
loc:@fc6/biases*
dtype0*
_output_shapes
:
І
'fc6/biases/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@fc6/biases*
dtype0*
_output_shapes
: 
█
!fc6/biases/Adam/Initializer/zerosFill1fc6/biases/Adam/Initializer/zeros/shape_as_tensor'fc6/biases/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@fc6/biases*
_output_shapes
:X
џ
fc6/biases/Adam
VariableV2*
shape:X*
dtype0*
_output_shapes
:X*
shared_name *
_class
loc:@fc6/biases*
	container 
┴
fc6/biases/Adam/AssignAssignfc6/biases/Adam!fc6/biases/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@fc6/biases*
validate_shape(*
_output_shapes
:X
u
fc6/biases/Adam/readIdentityfc6/biases/Adam*
T0*
_class
loc:@fc6/biases*
_output_shapes
:X
ю
3fc6/biases/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:X*
_class
loc:@fc6/biases*
dtype0*
_output_shapes
:
Ї
)fc6/biases/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@fc6/biases*
dtype0*
_output_shapes
: 
р
#fc6/biases/Adam_1/Initializer/zerosFill3fc6/biases/Adam_1/Initializer/zeros/shape_as_tensor)fc6/biases/Adam_1/Initializer/zeros/Const*
_output_shapes
:X*
T0*

index_type0*
_class
loc:@fc6/biases
ю
fc6/biases/Adam_1
VariableV2*
shared_name *
_class
loc:@fc6/biases*
	container *
shape:X*
dtype0*
_output_shapes
:X
К
fc6/biases/Adam_1/AssignAssignfc6/biases/Adam_1#fc6/biases/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@fc6/biases*
validate_shape(*
_output_shapes
:X
y
fc6/biases/Adam_1/readIdentityfc6/biases/Adam_1*
_output_shapes
:X*
T0*
_class
loc:@fc6/biases
W
Adam/learning_rateConst*
valueB
 *иQ9*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wЙ?
Q
Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
■
#Adam/update_conv1/weights/ApplyAdam	ApplyAdamconv1/weightsconv1/weights/Adamconv1/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/Kelz/conv1/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
: *
use_locking( *
T0* 
_class
loc:@conv1/weights
Ь
"Adam/update_conv1/biases/ApplyAdam	ApplyAdamconv1/biasesconv1/biases/Adamconv1/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/Kelz/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv1/biases*
use_nesterov( *
_output_shapes
: *
use_locking( 
■
#Adam/update_conv2/weights/ApplyAdam	ApplyAdamconv2/weightsconv2/weights/Adamconv2/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/Kelz/conv2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@conv2/weights*
use_nesterov( *&
_output_shapes
:  
Д
*Adam/update_conv2/BatchNorm/beta/ApplyAdam	ApplyAdamconv2/BatchNorm/betaconv2/BatchNorm/beta/Adamconv2/BatchNorm/beta/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMgradients/Kelz/conv2/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency_2*
use_locking( *
T0*'
_class
loc:@conv2/BatchNorm/beta*
use_nesterov( *
_output_shapes
: 
■
#Adam/update_conv3/weights/ApplyAdam	ApplyAdamconv3/weightsconv3/weights/Adamconv3/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/Kelz/conv3/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
: @*
use_locking( *
T0* 
_class
loc:@conv3/weights
Ь
"Adam/update_conv3/biases/ApplyAdam	ApplyAdamconv3/biasesconv3/biases/Adamconv3/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/Kelz/conv3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@conv3/biases*
use_nesterov( *
_output_shapes
:@
Ж
!Adam/update_fc5/weights/ApplyAdam	ApplyAdamfc5/weightsfc5/weights/Adamfc5/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/Kelz/fc5/Tensordot/transpose_1_grad/transpose*
T0*
_class
loc:@fc5/weights*
use_nesterov( * 
_output_shapes
:
ђ,ђ*
use_locking( 
с
 Adam/update_fc5/biases/ApplyAdam	ApplyAdam
fc5/biasesfc5/biases/Adamfc5/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon:gradients/Kelz/fc5/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@fc5/biases*
use_nesterov( *
_output_shapes	
:ђ
ж
!Adam/update_fc6/weights/ApplyAdam	ApplyAdamfc6/weightsfc6/weights/Adamfc6/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/Kelz/fc6/Tensordot/transpose_1_grad/transpose*
T0*
_class
loc:@fc6/weights*
use_nesterov( *
_output_shapes
:	ђX*
use_locking( 
Р
 Adam/update_fc6/biases/ApplyAdam	ApplyAdam
fc6/biasesfc6/biases/Adamfc6/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon:gradients/Kelz/fc6/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@fc6/biases*
use_nesterov( *
_output_shapes
:X
Т
Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_conv1/weights/ApplyAdam#^Adam/update_conv1/biases/ApplyAdam$^Adam/update_conv2/weights/ApplyAdam+^Adam/update_conv2/BatchNorm/beta/ApplyAdam$^Adam/update_conv3/weights/ApplyAdam#^Adam/update_conv3/biases/ApplyAdam"^Adam/update_fc5/weights/ApplyAdam!^Adam/update_fc5/biases/ApplyAdam"^Adam/update_fc6/weights/ApplyAdam!^Adam/update_fc6/biases/ApplyAdam*
T0*
_class
loc:@conv1/biases*
_output_shapes
: 
Ќ
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
У

Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_conv1/weights/ApplyAdam#^Adam/update_conv1/biases/ApplyAdam$^Adam/update_conv2/weights/ApplyAdam+^Adam/update_conv2/BatchNorm/beta/ApplyAdam$^Adam/update_conv3/weights/ApplyAdam#^Adam/update_conv3/biases/ApplyAdam"^Adam/update_fc5/weights/ApplyAdam!^Adam/update_fc5/biases/ApplyAdam"^Adam/update_fc6/weights/ApplyAdam!^Adam/update_fc6/biases/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@conv1/biases
Џ
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
А
AdamNoOp$^Adam/update_conv1/weights/ApplyAdam#^Adam/update_conv1/biases/ApplyAdam$^Adam/update_conv2/weights/ApplyAdam+^Adam/update_conv2/BatchNorm/beta/ApplyAdam$^Adam/update_conv3/weights/ApplyAdam#^Adam/update_conv3/biases/ApplyAdam"^Adam/update_fc5/weights/ApplyAdam!^Adam/update_fc5/biases/ApplyAdam"^Adam/update_fc6/weights/ApplyAdam!^Adam/update_fc6/biases/ApplyAdam^Adam/Assign^Adam/Assign_1
Ъ
initNoOp^conv1/weights/Assign^conv1/biases/Assign^conv2/weights/Assign^conv2/BatchNorm/beta/Assign#^conv2/BatchNorm/moving_mean/Assign'^conv2/BatchNorm/moving_variance/Assign^conv3/weights/Assign^conv3/biases/Assign^fc5/weights/Assign^fc5/biases/Assign^fc6/weights/Assign^fc6/biases/Assign^beta1_power/Assign^beta2_power/Assign^conv1/weights/Adam/Assign^conv1/weights/Adam_1/Assign^conv1/biases/Adam/Assign^conv1/biases/Adam_1/Assign^conv2/weights/Adam/Assign^conv2/weights/Adam_1/Assign!^conv2/BatchNorm/beta/Adam/Assign#^conv2/BatchNorm/beta/Adam_1/Assign^conv3/weights/Adam/Assign^conv3/weights/Adam_1/Assign^conv3/biases/Adam/Assign^conv3/biases/Adam_1/Assign^fc5/weights/Adam/Assign^fc5/weights/Adam_1/Assign^fc5/biases/Adam/Assign^fc5/biases/Adam_1/Assign^fc6/weights/Adam/Assign^fc6/weights/Adam_1/Assign^fc6/biases/Adam/Assign^fc6/biases/Adam_1/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
т
save/SaveV2/tensor_namesConst*ў
valueјBІ"Bbeta1_powerBbeta2_powerBconv1/biasesBconv1/biases/AdamBconv1/biases/Adam_1Bconv1/weightsBconv1/weights/AdamBconv1/weights/Adam_1Bconv2/BatchNorm/betaBconv2/BatchNorm/beta/AdamBconv2/BatchNorm/beta/Adam_1Bconv2/BatchNorm/moving_meanBconv2/BatchNorm/moving_varianceBconv2/weightsBconv2/weights/AdamBconv2/weights/Adam_1Bconv3/biasesBconv3/biases/AdamBconv3/biases/Adam_1Bconv3/weightsBconv3/weights/AdamBconv3/weights/Adam_1B
fc5/biasesBfc5/biases/AdamBfc5/biases/Adam_1Bfc5/weightsBfc5/weights/AdamBfc5/weights/Adam_1B
fc6/biasesBfc6/biases/AdamBfc6/biases/Adam_1Bfc6/weightsBfc6/weights/AdamBfc6/weights/Adam_1*
dtype0*
_output_shapes
:"
Д
save/SaveV2/shape_and_slicesConst*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:"
ј
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerconv1/biasesconv1/biases/Adamconv1/biases/Adam_1conv1/weightsconv1/weights/Adamconv1/weights/Adam_1conv2/BatchNorm/betaconv2/BatchNorm/beta/Adamconv2/BatchNorm/beta/Adam_1conv2/BatchNorm/moving_meanconv2/BatchNorm/moving_varianceconv2/weightsconv2/weights/Adamconv2/weights/Adam_1conv3/biasesconv3/biases/Adamconv3/biases/Adam_1conv3/weightsconv3/weights/Adamconv3/weights/Adam_1
fc5/biasesfc5/biases/Adamfc5/biases/Adam_1fc5/weightsfc5/weights/Adamfc5/weights/Adam_1
fc6/biasesfc6/biases/Adamfc6/biases/Adam_1fc6/weightsfc6/weights/Adamfc6/weights/Adam_1*0
dtypes&
$2"
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
э
save/RestoreV2/tensor_namesConst"/device:CPU:0*ў
valueјBІ"Bbeta1_powerBbeta2_powerBconv1/biasesBconv1/biases/AdamBconv1/biases/Adam_1Bconv1/weightsBconv1/weights/AdamBconv1/weights/Adam_1Bconv2/BatchNorm/betaBconv2/BatchNorm/beta/AdamBconv2/BatchNorm/beta/Adam_1Bconv2/BatchNorm/moving_meanBconv2/BatchNorm/moving_varianceBconv2/weightsBconv2/weights/AdamBconv2/weights/Adam_1Bconv3/biasesBconv3/biases/AdamBconv3/biases/Adam_1Bconv3/weightsBconv3/weights/AdamBconv3/weights/Adam_1B
fc5/biasesBfc5/biases/AdamBfc5/biases/Adam_1Bfc5/weightsBfc5/weights/AdamBfc5/weights/Adam_1B
fc6/biasesBfc6/biases/AdamBfc6/biases/Adam_1Bfc6/weightsBfc6/weights/AdamBfc6/weights/Adam_1*
dtype0*
_output_shapes
:"
╣
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:"
К
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*ъ
_output_shapesІ
ѕ::::::::::::::::::::::::::::::::::*0
dtypes&
$2"
Ю
save/AssignAssignbeta1_powersave/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
А
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
д
save/Assign_2Assignconv1/biasessave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
Ф
save/Assign_3Assignconv1/biases/Adamsave/RestoreV2:3*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
: 
Г
save/Assign_4Assignconv1/biases/Adam_1save/RestoreV2:4*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@conv1/biases
┤
save/Assign_5Assignconv1/weightssave/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
╣
save/Assign_6Assignconv1/weights/Adamsave/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
: 
╗
save/Assign_7Assignconv1/weights/Adam_1save/RestoreV2:7*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@conv1/weights
Х
save/Assign_8Assignconv2/BatchNorm/betasave/RestoreV2:8*
use_locking(*
T0*'
_class
loc:@conv2/BatchNorm/beta*
validate_shape(*
_output_shapes
: 
╗
save/Assign_9Assignconv2/BatchNorm/beta/Adamsave/RestoreV2:9*
T0*'
_class
loc:@conv2/BatchNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(
┐
save/Assign_10Assignconv2/BatchNorm/beta/Adam_1save/RestoreV2:10*
T0*'
_class
loc:@conv2/BatchNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(
к
save/Assign_11Assignconv2/BatchNorm/moving_meansave/RestoreV2:11*
use_locking(*
T0*.
_class$
" loc:@conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
: 
╬
save/Assign_12Assignconv2/BatchNorm/moving_variancesave/RestoreV2:12*
use_locking(*
T0*2
_class(
&$loc:@conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
: 
Х
save/Assign_13Assignconv2/weightssave/RestoreV2:13*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0* 
_class
loc:@conv2/weights
╗
save/Assign_14Assignconv2/weights/Adamsave/RestoreV2:14*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
:  
й
save/Assign_15Assignconv2/weights/Adam_1save/RestoreV2:15*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
:  
е
save/Assign_16Assignconv3/biasessave/RestoreV2:16*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes
:@
Г
save/Assign_17Assignconv3/biases/Adamsave/RestoreV2:17*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes
:@
»
save/Assign_18Assignconv3/biases/Adam_1save/RestoreV2:18*
use_locking(*
T0*
_class
loc:@conv3/biases*
validate_shape(*
_output_shapes
:@
Х
save/Assign_19Assignconv3/weightssave/RestoreV2:19*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv3/weights
╗
save/Assign_20Assignconv3/weights/Adamsave/RestoreV2:20*
use_locking(*
T0* 
_class
loc:@conv3/weights*
validate_shape(*&
_output_shapes
: @
й
save/Assign_21Assignconv3/weights/Adam_1save/RestoreV2:21*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0* 
_class
loc:@conv3/weights
Ц
save/Assign_22Assign
fc5/biasessave/RestoreV2:22*
use_locking(*
T0*
_class
loc:@fc5/biases*
validate_shape(*
_output_shapes	
:ђ
ф
save/Assign_23Assignfc5/biases/Adamsave/RestoreV2:23*
T0*
_class
loc:@fc5/biases*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
г
save/Assign_24Assignfc5/biases/Adam_1save/RestoreV2:24*
use_locking(*
T0*
_class
loc:@fc5/biases*
validate_shape(*
_output_shapes	
:ђ
г
save/Assign_25Assignfc5/weightssave/RestoreV2:25*
validate_shape(* 
_output_shapes
:
ђ,ђ*
use_locking(*
T0*
_class
loc:@fc5/weights
▒
save/Assign_26Assignfc5/weights/Adamsave/RestoreV2:26*
use_locking(*
T0*
_class
loc:@fc5/weights*
validate_shape(* 
_output_shapes
:
ђ,ђ
│
save/Assign_27Assignfc5/weights/Adam_1save/RestoreV2:27*
use_locking(*
T0*
_class
loc:@fc5/weights*
validate_shape(* 
_output_shapes
:
ђ,ђ
ц
save/Assign_28Assign
fc6/biasessave/RestoreV2:28*
use_locking(*
T0*
_class
loc:@fc6/biases*
validate_shape(*
_output_shapes
:X
Е
save/Assign_29Assignfc6/biases/Adamsave/RestoreV2:29*
validate_shape(*
_output_shapes
:X*
use_locking(*
T0*
_class
loc:@fc6/biases
Ф
save/Assign_30Assignfc6/biases/Adam_1save/RestoreV2:30*
T0*
_class
loc:@fc6/biases*
validate_shape(*
_output_shapes
:X*
use_locking(
Ф
save/Assign_31Assignfc6/weightssave/RestoreV2:31*
T0*
_class
loc:@fc6/weights*
validate_shape(*
_output_shapes
:	ђX*
use_locking(
░
save/Assign_32Assignfc6/weights/Adamsave/RestoreV2:32*
use_locking(*
T0*
_class
loc:@fc6/weights*
validate_shape(*
_output_shapes
:	ђX
▓
save/Assign_33Assignfc6/weights/Adam_1save/RestoreV2:33*
use_locking(*
T0*
_class
loc:@fc6/weights*
validate_shape(*
_output_shapes
:	ђX
╬
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33"F«Ў