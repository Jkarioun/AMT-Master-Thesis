       БK"	  ђ`ЕеоAbrain.Event:2щ§7Рѕ     4Pp3	;Ю╝`ЕеоA"НЉ
ї
inputPlaceholder*
dtype0*9
_output_shapes'
%:#                  Я*.
shape%:#                  Я
ѕ
PlaceholderPlaceholder*
dtype0*4
_output_shapes"
 :                  X*)
shape :                  X
і
Placeholder_1Placeholder*
dtype0*4
_output_shapes"
 :                  X*)
shape :                  X
N
Placeholder_2Placeholder*
dtype0
*
_output_shapes
: *
shape: 
▒
2conv1_mod/weights/Initializer/random_uniform/shapeConst*%
valueB"             *$
_class
loc:@conv1_mod/weights*
dtype0*
_output_shapes
:
Џ
0conv1_mod/weights/Initializer/random_uniform/minConst*
valueB
 *ьнMЙ*$
_class
loc:@conv1_mod/weights*
dtype0*
_output_shapes
: 
Џ
0conv1_mod/weights/Initializer/random_uniform/maxConst*
valueB
 *ьнM>*$
_class
loc:@conv1_mod/weights*
dtype0*
_output_shapes
: 
Ч
:conv1_mod/weights/Initializer/random_uniform/RandomUniformRandomUniform2conv1_mod/weights/Initializer/random_uniform/shape*

seed**
T0*$
_class
loc:@conv1_mod/weights*
seed2*
dtype0*&
_output_shapes
: 
Р
0conv1_mod/weights/Initializer/random_uniform/subSub0conv1_mod/weights/Initializer/random_uniform/max0conv1_mod/weights/Initializer/random_uniform/min*
T0*$
_class
loc:@conv1_mod/weights*
_output_shapes
: 
Ч
0conv1_mod/weights/Initializer/random_uniform/mulMul:conv1_mod/weights/Initializer/random_uniform/RandomUniform0conv1_mod/weights/Initializer/random_uniform/sub*
T0*$
_class
loc:@conv1_mod/weights*&
_output_shapes
: 
Ь
,conv1_mod/weights/Initializer/random_uniformAdd0conv1_mod/weights/Initializer/random_uniform/mul0conv1_mod/weights/Initializer/random_uniform/min*
T0*$
_class
loc:@conv1_mod/weights*&
_output_shapes
: 
╗
conv1_mod/weights
VariableV2*$
_class
loc:@conv1_mod/weights*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name 
с
conv1_mod/weights/AssignAssignconv1_mod/weights,conv1_mod/weights/Initializer/random_uniform*
use_locking(*
T0*$
_class
loc:@conv1_mod/weights*
validate_shape(*&
_output_shapes
: 
ї
conv1_mod/weights/readIdentityconv1_mod/weights*&
_output_shapes
: *
T0*$
_class
loc:@conv1_mod/weights
А
2conv1_mod/biases/Initializer/zeros/shape_as_tensorConst*
valueB: *#
_class
loc:@conv1_mod/biases*
dtype0*
_output_shapes
:
њ
(conv1_mod/biases/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *#
_class
loc:@conv1_mod/biases
С
"conv1_mod/biases/Initializer/zerosFill2conv1_mod/biases/Initializer/zeros/shape_as_tensor(conv1_mod/biases/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@conv1_mod/biases*
_output_shapes
: 
А
conv1_mod/biases
VariableV2*
dtype0*
_output_shapes
: *
shared_name *#
_class
loc:@conv1_mod/biases*
	container *
shape: 
╩
conv1_mod/biases/AssignAssignconv1_mod/biases"conv1_mod/biases/Initializer/zeros*
T0*#
_class
loc:@conv1_mod/biases*
validate_shape(*
_output_shapes
: *
use_locking(
}
conv1_mod/biases/readIdentityconv1_mod/biases*
T0*#
_class
loc:@conv1_mod/biases*
_output_shapes
: 
h
conv1_mod/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
в
conv1_mod/Conv2DConv2Dinputconv1_mod/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*9
_output_shapes'
%:#                  Я 
а
conv1_mod/BiasAddBiasAddconv1_mod/Conv2Dconv1_mod/biases/read*
T0*
data_formatNHWC*9
_output_shapes'
%:#                  Я 
m
conv1_mod/ReluReluconv1_mod/BiasAdd*
T0*9
_output_shapes'
%:#                  Я 
Ё
harmonic_layer/ConstConst*9
value0B."                              *
dtype0*
_output_shapes

:
ћ
harmonic_layer/PadPadconv1_mod/Reluharmonic_layer/Const*
T0*
	Tpaddings0*9
_output_shapes'
%:#                  р 
v
harmonic_layer/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Е
harmonic_layer/transpose	Transposeharmonic_layer/Padharmonic_layer/transpose/perm*
T0*9
_output_shapes'
%:#р                   *
Tperm0
ЭM
!harmonic_layer/Reordering/indicesConst*ЮM
valueЊMBљM	а"ђM`  `  `      0   L   o   `  `  `     1   M   p   `  `  `     2   N   q   `  `  `     3   O   r   `  `  `     4   P   s   `  `  `     5   Q   t   `  `  `     6   R   u   `  `  `     7   S   v   `  `  `     8   T   w   `  `  `  	   9   U   x   `  `  `  
   :   V   y   `  `  `     ;   W   z   `  `  `     <   X   {   `  `  `     =   Y   |   `  `  `     >   Z   }   `  `  `     ?   [   ~   `  `  `     @   \      `  `  `     A   ]   ђ   `  `  `     B   ^   Ђ   `  `  `     C   _   ѓ   `  `  `     D   `   Ѓ   `  `  `     E   a   ё   `  `  `     F   b   Ё   `  `  `     G   c   є   `  `  `     H   d   Є   `  `  `     I   e   ѕ   `  `  `     J   f   Ѕ   `  `  `     K   g   і   `  `  `     L   h   І   `  `  `     M   i   ї   `  `  `     N   j   Ї   `  `  `     O   k   ј   `  `  `      P   l   Ј   `  `  `  !   Q   m   љ   `  `  `  "   R   n   Љ   `  `  `  #   S   o   њ   `  `  `  $   T   p   Њ   `  `  `  %   U   q   ћ   `  `  `  &   V   r   Ћ   `  `  `  '   W   s   ќ   `  `  `  (   X   t   Ќ   `  `  `  )   Y   u   ў   `  `  `  *   Z   v   Ў   `  `  `  +   [   w   џ   `  `  `  ,   \   x   Џ   `  `  `  -   ]   y   ю   `  `  `  .   ^   z   Ю   `  `  `  /   _   {   ъ   `  `      0   `   |   Ъ   `  `     1   a   }   а   `  `     2   b   ~   А   `  `     3   c      б   `  `     4   d   ђ   Б   `  `     5   e   Ђ   ц   `  `     6   f   ѓ   Ц   `  `     7   g   Ѓ   д   `  `     8   h   ё   Д   `  `  	   9   i   Ё   е   `  `  
   :   j   є   Е   `  `     ;   k   Є   ф   `  `     <   l   ѕ   Ф   `  `     =   m   Ѕ   г   `  `     >   n   і   Г   `  `     ?   o   І   «   `  `     @   p   ї   »   `  `     A   q   Ї   ░   `  `     B   r   ј   ▒   `  `     C   s   Ј   ▓   `  `     D   t   љ   │   `  `     E   u   Љ   ┤   `  `     F   v   њ   х   `  `     G   w   Њ   Х   `  `     H   x   ћ   и   `  `     I   y   Ћ   И   `  `     J   z   ќ   ╣   `  `     K   {   Ќ   ║   `         L   |   ў   ╗   `        M   }   Ў   ╝   `        N   ~   џ   й   `        O      Џ   Й   `         P   ђ   ю   ┐   `     !   Q   Ђ   Ю   └   `     "   R   ѓ   ъ   ┴   `     #   S   Ѓ   Ъ   ┬   `     $   T   ё   а   ├   `  	   %   U   Ё   А   ─   `  
   &   V   є   б   ┼   `     '   W   Є   Б   к   `     (   X   ѕ   ц   К   `     )   Y   Ѕ   Ц   ╚   `     *   Z   і   д   ╔   `     +   [   І   Д   ╩   `     ,   \   ї   е   ╦   `     -   ]   Ї   Е   ╠   `     .   ^   ј   ф   ═   `     /   _   Ј   Ф   ╬   `     0   `   љ   г   ¤   `     1   a   Љ   Г   л   `     2   b   њ   «   Л   `     3   c   Њ   »   м   `     4   d   ћ   ░   М   `     5   e   Ћ   ▒   н   `     6   f   ќ   ▓   Н   `     7   g   Ќ   │   о   `     8   h   ў   ┤   О   `     9   i   Ў   х   п   `     :   j   џ   Х   ┘   `     ;   k   Џ   и   ┌   `      <   l   ю   И   █   `  !   =   m   Ю   ╣   ▄   `  "   >   n   ъ   ║   П       #   ?   o   Ъ   ╗   я      $   @   p   а   ╝   ▀      %   A   q   А   й   Я      &   B   r   б   Й   р      '   C   s   Б   ┐   Р      (   D   t   ц   └   с      )   E   u   Ц   ┴   С      *   F   v   д   ┬   т      +   G   w   Д   ├   Т   	   ,   H   x   е   ─   у   
   -   I   y   Е   ┼   У      .   J   z   ф   к   ж      /   K   {   Ф   К   Ж      0   L   |   г   ╚   в      1   M   }   Г   ╔   В      2   N   ~   «   ╩   ь      3   O      »   ╦   Ь      4   P   ђ   ░   ╠   №      5   Q   Ђ   ▒   ═   ­      6   R   ѓ   ▓   ╬   ы      7   S   Ѓ   │   ¤   Ы      8   T   ё   ┤   л   з      9   U   Ё   х   Л   З      :   V   є   Х   м   ш      ;   W   Є   и   М   Ш      <   X   ѕ   И   н   э      =   Y   Ѕ   ╣   Н   Э      >   Z   і   ║   о   щ      ?   [   І   ╗   О   Щ      @   \   ї   ╝   п   ч      A   ]   Ї   й   ┘   Ч      B   ^   ј   Й   ┌   §       C   _   Ј   ┐   █   ■   !   D   `   љ   └   ▄       "   E   a   Љ   ┴   П      #   F   b   њ   ┬   я     $   G   c   Њ   ├   ▀     %   H   d   ћ   ─   Я     &   I   e   Ћ   ┼   р     '   J   f   ќ   к   Р     (   K   g   Ќ   К   с     )   L   h   ў   ╚   С     *   M   i   Ў   ╔   т     +   N   j   џ   ╩   Т   	  ,   O   k   Џ   ╦   у   
  -   P   l   ю   ╠   У     .   Q   m   Ю   ═   ж     /   R   n   ъ   ╬   Ж     0   S   o   Ъ   ¤   в     1   T   p   а   л   В     2   U   q   А   Л   ь     3   V   r   б   м   Ь     4   W   s   Б   М   №     5   X   t   ц   н   ­     6   Y   u   Ц   Н   ы     7   Z   v   д   о   Ы     8   [   w   Д   О   з     9   \   x   е   п   З     :   ]   y   Е   ┘   ш     ;   ^   z   ф   ┌   Ш     <   _   {   Ф   █   э     =   `   |   г   ▄   Э     >   a   }   Г   П   щ     ?   b   ~   «   я   Щ     @   c      »   ▀   ч     A   d   ђ   ░   Я   Ч     B   e   Ђ   ▒   р   §      C   f   ѓ   ▓   Р   ■   !  D   g   Ѓ   │   с       "  E   h   ё   ┤   С      #  F   i   Ё   х   т     $  G   j   є   Х   Т     %  H   k   Є   и   у     &  I   l   ѕ   И   У     '  J   m   Ѕ   ╣   ж     (  K   n   і   ║   Ж     )  L   o   І   ╗   в     *  M   p   ї   ╝   В     +  N   q   Ї   й   ь   	  ,  O   r   ј   Й   Ь   
  -  P   s   Ј   ┐   №     .  Q   t   љ   └   ­     /  R   u   Љ   ┴   ы     0  S   v   њ   ┬   Ы     1  T   w   Њ   ├   з     2  U   x   ћ   ─   З     3  V   y   Ћ   ┼   ш     4  W   z   ќ   к   Ш     5  X   {   Ќ   К   э     6  Y   |   ў   ╚   Э     7  Z   }   Ў   ╔   щ     8  [   ~   џ   ╩   Щ     9  \      Џ   ╦   ч     :  ]   ђ   ю   ╠   Ч     ;  ^   Ђ   Ю   ═   §     <  _   ѓ   ъ   ╬   ■     =  `   Ѓ   Ъ   ¤         >  a   ё   а   л        ?  b   Ё   А   Л       @  c   є   б   м       A  d   Є   Б   М       B  e   ѕ   ц   н        C  f   Ѕ   Ц   Н     !  D  g   і   д   о     "  E  h   І   Д   О     #  F  i   ї   е   п     $  G  j   Ї   Е   ┘   	  %  H  k   ј   ф   ┌   
  &  I  l   Ј   Ф   █     '  J  m   љ   г   ▄     (  K  n   Љ   Г   П     )  L  o   њ   «   я     *  M  p   Њ   »   ▀     +  N  q   ћ   ░   Я     ,  O  r   Ћ   ▒   р     -  P  s   ќ   ▓   Р     .  Q  t   Ќ   │   с     /  R  u   ў   ┤   С     0  S  v   Ў   х   т     1  T  w   џ   Х   Т     2  U  x   Џ   и   у     3  V  y   ю   И   У     4  W  z   Ю   ╣   ж     5  X  {   ъ   ║   Ж     6  Y  |   Ъ   ╗   в     7  Z  }   а   ╝   В     8  [  ~   А   й   ь     9  \     б   Й   Ь     :  ]  ђ   Б   ┐   №     ;  ^  Ђ   ц   └   ­      <  _  ѓ   Ц   ┴   ы   !  =  `  Ѓ   д   ┬   Ы   "  >  `  ё   Д   ├   з   #  ?  `  Ё   е   ─   З   $  @  `  є   Е   ┼   ш   %  A  `  Є   ф   к   Ш   &  B  `  ѕ   Ф   К   э   '  C  `  Ѕ   г   ╚   Э   (  D  `  і   Г   ╔   щ   )  E  `  І   «   ╩   Щ   *  F  `  ї   »   ╦   ч   +  G  `  Ї   ░   ╠   Ч   ,  H  `  ј   ▒   ═   §   -  I  `  Ј   ▓   ╬   ■   .  J  `  љ   │   ¤       /  K  `  Љ   ┤   л      0  L  `  њ   х   Л     1  M  `  Њ   Х   м     2  N  `  ћ   и   М     3  O  `  Ћ   И   н     4  P  `  ќ   ╣   Н     5  Q  `  Ќ   ║   о     6  R  `  ў   ╗   О     7  S  `  Ў   ╝   п     8  T  `  џ   й   ┘   	  9  U  `  Џ   Й   ┌   
  :  V  `  ю   ┐   █     ;  W  `  Ю   └   ▄     <  X  `  ъ   ┴   П     =  Y  `  Ъ   ┬   я     >  Z  `  а   ├   ▀     ?  [  `  А   ─   Я     @  \  `  б   ┼   р     A  ]  `  Б   к   Р     B  ^  `  ц   К   с     C  _  `  Ц   ╚   С     D  `  `  д   ╔   т     E  `  `  Д   ╩   Т     F  `  `  е   ╦   у     G  `  `  Е   ╠   У     H  `  `  ф   ═   ж     I  `  `  Ф   ╬   Ж     J  `  `  г   ¤   в     K  `  `  Г   л   В     L  `  `  «   Л   ь     M  `  `  »   м   Ь     N  `  `  ░   М   №     O  `  `  ▒   н   ­      P  `  `  ▓   Н   ы   !  Q  `  `  │   о   Ы   "  R  `  `  ┤   О   з   #  S  `  `  х   п   З   $  T  `  `  Х   ┘   ш   %  U  `  `  и   ┌   Ш   &  V  `  `  И   █   э   '  W  `  `  ╣   ▄   Э   (  X  `  `  ║   П   щ   )  Y  `  `  ╗   я   Щ   *  Z  `  `  ╝   ▀   ч   +  [  `  `  й   Я   Ч   ,  \  `  `  Й   р   §   -  ]  `  `  ┐   Р   ■   .  ^  `  `  └   с       /  _  `  `  ┴   С      0  `  `  `  ┬   т     1  `  `  `  ├   Т     2  `  `  `  ─   у     3  `  `  `  ┼   У     4  `  `  `  к   ж     5  `  `  `  К   Ж     6  `  `  `  ╚   в     7  `  `  `  ╔   В     8  `  `  `  ╩   ь   	  9  `  `  `  ╦   Ь   
  :  `  `  `  ╠   №     ;  `  `  `  ═   ­     <  `  `  `  ╬   ы     =  `  `  `  ¤   Ы     >  `  `  `  л   з     ?  `  `  `  Л   З     @  `  `  `  м   ш     A  `  `  `  М   Ш     B  `  `  `  н   э     C  `  `  `  Н   Э     D  `  `  `  о   щ     E  `  `  `  О   Щ     F  `  `  `  п   ч     G  `  `  `  ┘   Ч     H  `  `  `  ┌   §     I  `  `  `  █   ■     J  `  `  `  ▄         K  `  `  `  П        L  `  `  `  я       M  `  `  `  ▀       N  `  `  `  Я       O  `  `  `  р        P  `  `  `  Р     !  Q  `  `  `  с     "  R  `  `  `  С     #  S  `  `  `  т     $  T  `  `  `  Т   	  %  U  `  `  `  у   
  &  V  `  `  `  У     '  W  `  `  `  ж     (  X  `  `  `  Ж     )  Y  `  `  `  в     *  Z  `  `  `  В     +  [  `  `  `  ь     ,  \  `  `  `  Ь     -  ]  `  `  `  №     .  ^  `  `  `  ­     /  _  `  `  `  *
dtype0*
_output_shapes
:	а
╝
harmonic_layer/ReorderingGatherNdharmonic_layer/transpose!harmonic_layer/Reordering/indices*
Tparams0*9
_output_shapes'
%:#а                   *
Tindices0
x
harmonic_layer/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
┤
harmonic_layer/transpose_1	Transposeharmonic_layer/Reorderingharmonic_layer/transpose_1/perm*9
_output_shapes'
%:#                  а *
Tperm0*
T0
▒
2conv2_mod/weights/Initializer/random_uniform/shapeConst*%
valueB"              *$
_class
loc:@conv2_mod/weights*
dtype0*
_output_shapes
:
Џ
0conv2_mod/weights/Initializer/random_uniform/minConst*
valueB
 *Јё┴й*$
_class
loc:@conv2_mod/weights*
dtype0*
_output_shapes
: 
Џ
0conv2_mod/weights/Initializer/random_uniform/maxConst*
valueB
 *Јё┴=*$
_class
loc:@conv2_mod/weights*
dtype0*
_output_shapes
: 
Ч
:conv2_mod/weights/Initializer/random_uniform/RandomUniformRandomUniform2conv2_mod/weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:  *

seed**
T0*$
_class
loc:@conv2_mod/weights*
seed2#
Р
0conv2_mod/weights/Initializer/random_uniform/subSub0conv2_mod/weights/Initializer/random_uniform/max0conv2_mod/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@conv2_mod/weights
Ч
0conv2_mod/weights/Initializer/random_uniform/mulMul:conv2_mod/weights/Initializer/random_uniform/RandomUniform0conv2_mod/weights/Initializer/random_uniform/sub*&
_output_shapes
:  *
T0*$
_class
loc:@conv2_mod/weights
Ь
,conv2_mod/weights/Initializer/random_uniformAdd0conv2_mod/weights/Initializer/random_uniform/mul0conv2_mod/weights/Initializer/random_uniform/min*&
_output_shapes
:  *
T0*$
_class
loc:@conv2_mod/weights
╗
conv2_mod/weights
VariableV2*
	container *
shape:  *
dtype0*&
_output_shapes
:  *
shared_name *$
_class
loc:@conv2_mod/weights
с
conv2_mod/weights/AssignAssignconv2_mod/weights,conv2_mod/weights/Initializer/random_uniform*
use_locking(*
T0*$
_class
loc:@conv2_mod/weights*
validate_shape(*&
_output_shapes
:  
ї
conv2_mod/weights/readIdentityconv2_mod/weights*
T0*$
_class
loc:@conv2_mod/weights*&
_output_shapes
:  
w
&harmonic_layer/conv2_mod/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
љ
harmonic_layer/conv2_mod/Conv2DConv2Dharmonic_layer/transpose_1conv2_mod/weights/read*9
_output_shapes'
%:#                  Я *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
u
(harmonic_layer/conv2_mod/BatchNorm/ConstConst*
valueB *  ђ?*
dtype0*
_output_shapes
: 
▒
:conv2_mod/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB: *+
_class!
loc:@conv2_mod/BatchNorm/beta
б
0conv2_mod/BatchNorm/beta/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@conv2_mod/BatchNorm/beta*
dtype0*
_output_shapes
: 
ё
*conv2_mod/BatchNorm/beta/Initializer/zerosFill:conv2_mod/BatchNorm/beta/Initializer/zeros/shape_as_tensor0conv2_mod/BatchNorm/beta/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@conv2_mod/BatchNorm/beta*
_output_shapes
: 
▒
conv2_mod/BatchNorm/beta
VariableV2*
shared_name *+
_class!
loc:@conv2_mod/BatchNorm/beta*
	container *
shape: *
dtype0*
_output_shapes
: 
Ж
conv2_mod/BatchNorm/beta/AssignAssignconv2_mod/BatchNorm/beta*conv2_mod/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@conv2_mod/BatchNorm/beta*
validate_shape(*
_output_shapes
: 
Ћ
conv2_mod/BatchNorm/beta/readIdentityconv2_mod/BatchNorm/beta*
_output_shapes
: *
T0*+
_class!
loc:@conv2_mod/BatchNorm/beta
┐
Aconv2_mod/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB: *2
_class(
&$loc:@conv2_mod/BatchNorm/moving_mean
░
7conv2_mod/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *2
_class(
&$loc:@conv2_mod/BatchNorm/moving_mean*
dtype0*
_output_shapes
: 
а
1conv2_mod/BatchNorm/moving_mean/Initializer/zerosFillAconv2_mod/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor7conv2_mod/BatchNorm/moving_mean/Initializer/zeros/Const*
T0*

index_type0*2
_class(
&$loc:@conv2_mod/BatchNorm/moving_mean*
_output_shapes
: 
┐
conv2_mod/BatchNorm/moving_mean
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *2
_class(
&$loc:@conv2_mod/BatchNorm/moving_mean*
	container 
є
&conv2_mod/BatchNorm/moving_mean/AssignAssignconv2_mod/BatchNorm/moving_mean1conv2_mod/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*2
_class(
&$loc:@conv2_mod/BatchNorm/moving_mean
ф
$conv2_mod/BatchNorm/moving_mean/readIdentityconv2_mod/BatchNorm/moving_mean*
_output_shapes
: *
T0*2
_class(
&$loc:@conv2_mod/BatchNorm/moving_mean
к
Dconv2_mod/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB: *6
_class,
*(loc:@conv2_mod/BatchNorm/moving_variance*
dtype0*
_output_shapes
:
и
:conv2_mod/BatchNorm/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ђ?*6
_class,
*(loc:@conv2_mod/BatchNorm/moving_variance*
dtype0*
_output_shapes
: 
Г
4conv2_mod/BatchNorm/moving_variance/Initializer/onesFillDconv2_mod/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor:conv2_mod/BatchNorm/moving_variance/Initializer/ones/Const*
T0*

index_type0*6
_class,
*(loc:@conv2_mod/BatchNorm/moving_variance*
_output_shapes
: 
К
#conv2_mod/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes
: *
shared_name *6
_class,
*(loc:@conv2_mod/BatchNorm/moving_variance*
	container *
shape: 
Ћ
*conv2_mod/BatchNorm/moving_variance/AssignAssign#conv2_mod/BatchNorm/moving_variance4conv2_mod/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*6
_class,
*(loc:@conv2_mod/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
: 
Х
(conv2_mod/BatchNorm/moving_variance/readIdentity#conv2_mod/BatchNorm/moving_variance*
T0*6
_class,
*(loc:@conv2_mod/BatchNorm/moving_variance*
_output_shapes
: 
m
*harmonic_layer/conv2_mod/BatchNorm/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
m
*harmonic_layer/conv2_mod/BatchNorm/Const_2Const*
valueB *
dtype0*
_output_shapes
: 
Ю
1harmonic_layer/conv2_mod/BatchNorm/FusedBatchNormFusedBatchNormharmonic_layer/conv2_mod/Conv2D(harmonic_layer/conv2_mod/BatchNorm/Constconv2_mod/BatchNorm/beta/read*harmonic_layer/conv2_mod/BatchNorm/Const_1*harmonic_layer/conv2_mod/BatchNorm/Const_2*
data_formatNHWC*Q
_output_shapes?
=:#                  Я : : : : *
is_training(*
epsilon%oЃ:*
T0
o
*harmonic_layer/conv2_mod/BatchNorm/Const_3Const*
valueB
 *oЃ:*
dtype0*
_output_shapes
: 
й
7harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg/readIdentityconv2_mod/BatchNorm/moving_mean*
T0*2
_class(
&$loc:@conv2_mod/BatchNorm/moving_mean*
_output_shapes
: 
ё
6harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg/SubSub7harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg/read3harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm:1*
T0*2
_class(
&$loc:@conv2_mod/BatchNorm/moving_mean*
_output_shapes
: 
Щ
6harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg/MulMul6harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg/Sub*harmonic_layer/conv2_mod/BatchNorm/Const_3*
_output_shapes
: *
T0*2
_class(
&$loc:@conv2_mod/BatchNorm/moving_mean
ё
2harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg	AssignSubconv2_mod/BatchNorm/moving_mean6harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg/Mul*
_output_shapes
: *
use_locking( *
T0*2
_class(
&$loc:@conv2_mod/BatchNorm/moving_mean
К
9harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg_1/readIdentity#conv2_mod/BatchNorm/moving_variance*
_output_shapes
: *
T0*6
_class,
*(loc:@conv2_mod/BatchNorm/moving_variance
ї
8harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg_1/SubSub9harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg_1/read3harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm:2*
_output_shapes
: *
T0*6
_class,
*(loc:@conv2_mod/BatchNorm/moving_variance
ѓ
8harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg_1/MulMul8harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg_1/Sub*harmonic_layer/conv2_mod/BatchNorm/Const_3*
_output_shapes
: *
T0*6
_class,
*(loc:@conv2_mod/BatchNorm/moving_variance
љ
4harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg_1	AssignSub#conv2_mod/BatchNorm/moving_variance8harmonic_layer/conv2_mod/BatchNorm/AssignMovingAvg_1/Mul*
_output_shapes
: *
use_locking( *
T0*6
_class,
*(loc:@conv2_mod/BatchNorm/moving_variance
ю
harmonic_layer/conv2_mod/ReluRelu1harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm*
T0*9
_output_shapes'
%:#                  Я 
м
pool2_mod/MaxPoolMaxPoolharmonic_layer/conv2_mod/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*9
_output_shapes'
%:#                  ░ 
c
dropout2_mod/cond/SwitchSwitchPlaceholder_2Placeholder_2*
T0
*
_output_shapes
: : 
c
dropout2_mod/cond/switch_tIdentitydropout2_mod/cond/Switch:1*
T0
*
_output_shapes
: 
a
dropout2_mod/cond/switch_fIdentitydropout2_mod/cond/Switch*
_output_shapes
: *
T0

U
dropout2_mod/cond/pred_idIdentityPlaceholder_2*
_output_shapes
: *
T0

Ё
#dropout2_mod/cond/dropout/keep_probConst^dropout2_mod/cond/switch_t*
valueB
 *  ђ>*
dtype0*
_output_shapes
: 
Є
dropout2_mod/cond/dropout/ShapeShape(dropout2_mod/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
ь
&dropout2_mod/cond/dropout/Shape/SwitchSwitchpool2_mod/MaxPooldropout2_mod/cond/pred_id*
T0*$
_class
loc:@pool2_mod/MaxPool*^
_output_shapesL
J:#                  ░ :#                  ░ 
ј
,dropout2_mod/cond/dropout/random_uniform/minConst^dropout2_mod/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
ј
,dropout2_mod/cond/dropout/random_uniform/maxConst^dropout2_mod/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
м
6dropout2_mod/cond/dropout/random_uniform/RandomUniformRandomUniformdropout2_mod/cond/dropout/Shape*
T0*
dtype0*9
_output_shapes'
%:#                  ░ *
seed2V*

seed*
░
,dropout2_mod/cond/dropout/random_uniform/subSub,dropout2_mod/cond/dropout/random_uniform/max,dropout2_mod/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
П
,dropout2_mod/cond/dropout/random_uniform/mulMul6dropout2_mod/cond/dropout/random_uniform/RandomUniform,dropout2_mod/cond/dropout/random_uniform/sub*
T0*9
_output_shapes'
%:#                  ░ 
¤
(dropout2_mod/cond/dropout/random_uniformAdd,dropout2_mod/cond/dropout/random_uniform/mul,dropout2_mod/cond/dropout/random_uniform/min*9
_output_shapes'
%:#                  ░ *
T0
и
dropout2_mod/cond/dropout/addAdd#dropout2_mod/cond/dropout/keep_prob(dropout2_mod/cond/dropout/random_uniform*
T0*9
_output_shapes'
%:#                  ░ 
І
dropout2_mod/cond/dropout/FloorFloordropout2_mod/cond/dropout/add*
T0*9
_output_shapes'
%:#                  ░ 
╗
dropout2_mod/cond/dropout/divRealDiv(dropout2_mod/cond/dropout/Shape/Switch:1#dropout2_mod/cond/dropout/keep_prob*9
_output_shapes'
%:#                  ░ *
T0
е
dropout2_mod/cond/dropout/mulMuldropout2_mod/cond/dropout/divdropout2_mod/cond/dropout/Floor*
T0*9
_output_shapes'
%:#                  ░ 
Ї
dropout2_mod/cond/IdentityIdentity!dropout2_mod/cond/Identity/Switch*
T0*9
_output_shapes'
%:#                  ░ 
У
!dropout2_mod/cond/Identity/SwitchSwitchpool2_mod/MaxPooldropout2_mod/cond/pred_id*^
_output_shapesL
J:#                  ░ :#                  ░ *
T0*$
_class
loc:@pool2_mod/MaxPool
ф
dropout2_mod/cond/MergeMergedropout2_mod/cond/Identitydropout2_mod/cond/dropout/mul*
T0*
N*;
_output_shapes)
':#                  ░ : 
Є
harmonic_layer_1/ConstConst*9
value0B."                              *
dtype0*
_output_shapes

:
А
harmonic_layer_1/PadPaddropout2_mod/cond/Mergeharmonic_layer_1/Const*
T0*
	Tpaddings0*9
_output_shapes'
%:#                  ▒ 
x
harmonic_layer_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
»
harmonic_layer_1/transpose	Transposeharmonic_layer_1/Padharmonic_layer_1/transpose/perm*9
_output_shapes'
%:#▒                   *
Tperm0*
T0
║'
#harmonic_layer_1/Reordering/indicesConst*П&
valueМ&Bл&	л	"└&░   ░   ░          &   8   ░   ░   ░         '   9   ░   ░   ░         (   :   ░   ░   ░         )   ;   ░   ░   ░         *   <   ░   ░   ░         +   =   ░   ░   ░         ,   >   ░   ░   ░         -   ?   ░   ░   ░          .   @   ░   ░   ░   	   !   /   A   ░   ░   ░   
   "   0   B   ░   ░   ░      #   1   C   ░   ░   ░      $   2   D   ░   ░   ░      %   3   E   ░   ░   ░      &   4   F   ░   ░   ░      '   5   G   ░   ░   ░      (   6   H   ░   ░   ░      )   7   I   ░   ░   ░      *   8   J   ░   ░   ░      +   9   K   ░   ░   ░      ,   :   L   ░   ░   ░      -   ;   M   ░   ░   ░      .   <   N   ░   ░   ░      /   =   O   ░   ░          0   >   P   ░   ░         1   ?   Q   ░   ░         2   @   R   ░   ░         3   A   S   ░   ░         4   B   T   ░   ░         5   C   U   ░   ░         6   D   V   ░   ░         7   E   W   ░   ░          8   F   X   ░   ░   	   !   9   G   Y   ░   ░   
   "   :   H   Z   ░   ░      #   ;   I   [   ░   ░      $   <   J   \   ░   ░      %   =   K   ]   ░          &   >   L   ^   ░         '   ?   M   _   ░         (   @   N   `   ░         )   A   O   a   ░         *   B   P   b   ░         +   C   Q   c   ░         ,   D   R   d   ░         -   E   S   e   ░         .   F   T   f   ░   	      /   G   U   g   ░   
      0   H   V   h   ░         1   I   W   i   ░         2   J   X   j   ░         3   K   Y   k   ░         4   L   Z   l   ░         5   M   [   m   ░         6   N   \   n   ░         7   O   ]   o              8   P   ^   p         !   9   Q   _   q         "   :   R   `   r         #   ;   S   a   s         $   <   T   b   t         %   =   U   c   u         &   >   V   d   v         '   ?   W   e   w         (   @   X   f   x   	      )   A   Y   g   y   
      *   B   Z   h   z         +   C   [   i   {         ,   D   \   j   |         -   E   ]   k   }          .   F   ^   l   ~      !   /   G   _   m         "   0   H   `   n   ђ      #   1   I   a   o   Ђ      $   2   J   b   p   ѓ      %   3   K   c   q   Ѓ      &   4   L   d   r   ё      '   5   M   e   s   Ё      (   6   N   f   t   є      )   7   O   g   u   Є      *   8   P   h   v   ѕ      +   9   Q   i   w   Ѕ      ,   :   R   j   x   і      -   ;   S   k   y   І      .   <   T   l   z   ї      /   =   U   m   {   Ї      0   >   V   n   |   ј      1   ?   W   o   }   Ј       2   @   X   p   ~   љ   !   3   A   Y   q      Љ   "   4   B   Z   r   ђ   њ   #   5   C   [   s   Ђ   Њ   $   6   D   \   t   ѓ   ћ   %   7   E   ]   u   Ѓ   Ћ   &   8   F   ^   v   ё   ќ   '   9   G   _   w   Ё   Ќ   (   :   H   `   x   є   ў   )   ;   I   a   y   Є   Ў   *   <   J   b   z   ѕ   џ   +   =   K   c   {   Ѕ   Џ   ,   >   L   d   |   і   ю   -   ?   M   e   }   І   Ю   .   @   N   f   ~   ї   ъ   /   A   O   g      Ї   Ъ   0   B   P   h   ђ   ј   а   1   C   Q   i   Ђ   Ј   А   2   D   R   j   ѓ   љ   б   3   E   S   k   Ѓ   Љ   Б   4   F   T   l   ё   њ   ц   5   G   U   m   Ё   Њ   Ц   6   H   V   n   є   ћ   д   7   I   W   o   Є   Ћ   Д   8   J   X   p   ѕ   ќ   е   9   K   Y   q   Ѕ   Ќ   Е   :   L   Z   r   і   ў   ф   ;   M   [   s   І   Ў   Ф   <   N   \   t   ї   џ   г   =   O   ]   u   Ї   Џ   Г   >   P   ^   v   ј   ю   «   ?   Q   _   w   Ј   Ю   »   @   R   `   x   љ   ъ   ░   A   S   a   y   Љ   Ъ   ░   B   T   b   z   њ   а   ░   C   U   c   {   Њ   А   ░   D   V   d   |   ћ   б   ░   E   W   e   }   Ћ   Б   ░   F   X   f   ~   ќ   ц   ░   G   Y   g      Ќ   Ц   ░   H   Z   h   ђ   ў   д   ░   I   [   i   Ђ   Ў   Д   ░   J   \   j   ѓ   џ   е   ░   K   ]   k   Ѓ   Џ   Е   ░   L   ^   l   ё   ю   ф   ░   M   _   m   Ё   Ю   Ф   ░   N   `   n   є   ъ   г   ░   O   a   o   Є   Ъ   Г   ░   P   b   p   ѕ   а   «   ░   Q   c   q   Ѕ   А   »   ░   R   d   r   і   б   ░   ░   S   e   s   І   Б   ░   ░   T   f   t   ї   ц   ░   ░   U   g   u   Ї   Ц   ░   ░   V   h   v   ј   д   ░   ░   W   i   w   Ј   Д   ░   ░   X   j   x   љ   е   ░   ░   Y   k   y   Љ   Е   ░   ░   Z   l   z   њ   ф   ░   ░   [   m   {   Њ   Ф   ░   ░   \   n   |   ћ   г   ░   ░   ]   o   }   Ћ   Г   ░   ░   ^   p   ~   ќ   «   ░   ░   _   q      Ќ   »   ░   ░   `   r   ђ   ў   ░   ░   ░   a   s   Ђ   Ў   ░   ░   ░   b   t   ѓ   џ   ░   ░   ░   c   u   Ѓ   Џ   ░   ░   ░   d   v   ё   ю   ░   ░   ░   e   w   Ё   Ю   ░   ░   ░   f   x   є   ъ   ░   ░   ░   g   y   Є   Ъ   ░   ░   ░   h   z   ѕ   а   ░   ░   ░   i   {   Ѕ   А   ░   ░   ░   j   |   і   б   ░   ░   ░   k   }   І   Б   ░   ░   ░   l   ~   ї   ц   ░   ░   ░   m      Ї   Ц   ░   ░   ░   n   ђ   ј   д   ░   ░   ░   o   Ђ   Ј   Д   ░   ░   ░   p   ѓ   љ   е   ░   ░   ░   q   Ѓ   Љ   Е   ░   ░   ░   r   ё   њ   ф   ░   ░   ░   s   Ё   Њ   Ф   ░   ░   ░   t   є   ћ   г   ░   ░   ░   u   Є   Ћ   Г   ░   ░   ░   v   ѕ   ќ   «   ░   ░   ░   w   Ѕ   Ќ   »   ░   ░   ░   *
dtype0*
_output_shapes
:	л	
┬
harmonic_layer_1/ReorderingGatherNdharmonic_layer_1/transpose#harmonic_layer_1/Reordering/indices*
Tparams0*9
_output_shapes'
%:#л	                   *
Tindices0
z
!harmonic_layer_1/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
║
harmonic_layer_1/transpose_1	Transposeharmonic_layer_1/Reordering!harmonic_layer_1/transpose_1/perm*
T0*9
_output_shapes'
%:#                  л	 *
Tperm0
▒
2conv3_mod/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"          @   *$
_class
loc:@conv3_mod/weights
Џ
0conv3_mod/weights/Initializer/random_uniform/minConst*
valueB
 *│ъй*$
_class
loc:@conv3_mod/weights*
dtype0*
_output_shapes
: 
Џ
0conv3_mod/weights/Initializer/random_uniform/maxConst*
valueB
 *│ъ=*$
_class
loc:@conv3_mod/weights*
dtype0*
_output_shapes
: 
Ч
:conv3_mod/weights/Initializer/random_uniform/RandomUniformRandomUniform2conv3_mod/weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: @*

seed**
T0*$
_class
loc:@conv3_mod/weights*
seed2l
Р
0conv3_mod/weights/Initializer/random_uniform/subSub0conv3_mod/weights/Initializer/random_uniform/max0conv3_mod/weights/Initializer/random_uniform/min*
T0*$
_class
loc:@conv3_mod/weights*
_output_shapes
: 
Ч
0conv3_mod/weights/Initializer/random_uniform/mulMul:conv3_mod/weights/Initializer/random_uniform/RandomUniform0conv3_mod/weights/Initializer/random_uniform/sub*
T0*$
_class
loc:@conv3_mod/weights*&
_output_shapes
: @
Ь
,conv3_mod/weights/Initializer/random_uniformAdd0conv3_mod/weights/Initializer/random_uniform/mul0conv3_mod/weights/Initializer/random_uniform/min*
T0*$
_class
loc:@conv3_mod/weights*&
_output_shapes
: @
╗
conv3_mod/weights
VariableV2*
dtype0*&
_output_shapes
: @*
shared_name *$
_class
loc:@conv3_mod/weights*
	container *
shape: @
с
conv3_mod/weights/AssignAssignconv3_mod/weights,conv3_mod/weights/Initializer/random_uniform*
use_locking(*
T0*$
_class
loc:@conv3_mod/weights*
validate_shape(*&
_output_shapes
: @
ї
conv3_mod/weights/readIdentityconv3_mod/weights*
T0*$
_class
loc:@conv3_mod/weights*&
_output_shapes
: @
А
2conv3_mod/biases/Initializer/zeros/shape_as_tensorConst*
valueB:@*#
_class
loc:@conv3_mod/biases*
dtype0*
_output_shapes
:
њ
(conv3_mod/biases/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@conv3_mod/biases*
dtype0*
_output_shapes
: 
С
"conv3_mod/biases/Initializer/zerosFill2conv3_mod/biases/Initializer/zeros/shape_as_tensor(conv3_mod/biases/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@conv3_mod/biases*
_output_shapes
:@
А
conv3_mod/biases
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *#
_class
loc:@conv3_mod/biases
╩
conv3_mod/biases/AssignAssignconv3_mod/biases"conv3_mod/biases/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@conv3_mod/biases*
validate_shape(*
_output_shapes
:@
}
conv3_mod/biases/readIdentityconv3_mod/biases*
T0*#
_class
loc:@conv3_mod/biases*
_output_shapes
:@
y
(harmonic_layer_1/conv3_mod/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ћ
!harmonic_layer_1/conv3_mod/Conv2DConv2Dharmonic_layer_1/transpose_1conv3_mod/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*9
_output_shapes'
%:#                  ░@
┬
"harmonic_layer_1/conv3_mod/BiasAddBiasAdd!harmonic_layer_1/conv3_mod/Conv2Dconv3_mod/biases/read*
data_formatNHWC*9
_output_shapes'
%:#                  ░@*
T0
Ј
harmonic_layer_1/conv3_mod/ReluRelu"harmonic_layer_1/conv3_mod/BiasAdd*
T0*9
_output_shapes'
%:#                  ░@
М
pool3_mod/MaxPoolMaxPoolharmonic_layer_1/conv3_mod/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*8
_output_shapes&
$:"                  X@
c
dropout3_mod/cond/SwitchSwitchPlaceholder_2Placeholder_2*
T0
*
_output_shapes
: : 
c
dropout3_mod/cond/switch_tIdentitydropout3_mod/cond/Switch:1*
_output_shapes
: *
T0

a
dropout3_mod/cond/switch_fIdentitydropout3_mod/cond/Switch*
T0
*
_output_shapes
: 
U
dropout3_mod/cond/pred_idIdentityPlaceholder_2*
T0
*
_output_shapes
: 
Ё
#dropout3_mod/cond/dropout/keep_probConst^dropout3_mod/cond/switch_t*
valueB
 *  ђ>*
dtype0*
_output_shapes
: 
Є
dropout3_mod/cond/dropout/ShapeShape(dropout3_mod/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
в
&dropout3_mod/cond/dropout/Shape/SwitchSwitchpool3_mod/MaxPooldropout3_mod/cond/pred_id*
T0*$
_class
loc:@pool3_mod/MaxPool*\
_output_shapesJ
H:"                  X@:"                  X@
ј
,dropout3_mod/cond/dropout/random_uniform/minConst^dropout3_mod/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
ј
,dropout3_mod/cond/dropout/random_uniform/maxConst^dropout3_mod/cond/switch_t*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
м
6dropout3_mod/cond/dropout/random_uniform/RandomUniformRandomUniformdropout3_mod/cond/dropout/Shape*
dtype0*8
_output_shapes&
$:"                  X@*
seed2Є*

seed**
T0
░
,dropout3_mod/cond/dropout/random_uniform/subSub,dropout3_mod/cond/dropout/random_uniform/max,dropout3_mod/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
▄
,dropout3_mod/cond/dropout/random_uniform/mulMul6dropout3_mod/cond/dropout/random_uniform/RandomUniform,dropout3_mod/cond/dropout/random_uniform/sub*8
_output_shapes&
$:"                  X@*
T0
╬
(dropout3_mod/cond/dropout/random_uniformAdd,dropout3_mod/cond/dropout/random_uniform/mul,dropout3_mod/cond/dropout/random_uniform/min*
T0*8
_output_shapes&
$:"                  X@
Х
dropout3_mod/cond/dropout/addAdd#dropout3_mod/cond/dropout/keep_prob(dropout3_mod/cond/dropout/random_uniform*
T0*8
_output_shapes&
$:"                  X@
і
dropout3_mod/cond/dropout/FloorFloordropout3_mod/cond/dropout/add*
T0*8
_output_shapes&
$:"                  X@
║
dropout3_mod/cond/dropout/divRealDiv(dropout3_mod/cond/dropout/Shape/Switch:1#dropout3_mod/cond/dropout/keep_prob*
T0*8
_output_shapes&
$:"                  X@
Д
dropout3_mod/cond/dropout/mulMuldropout3_mod/cond/dropout/divdropout3_mod/cond/dropout/Floor*
T0*8
_output_shapes&
$:"                  X@
ї
dropout3_mod/cond/IdentityIdentity!dropout3_mod/cond/Identity/Switch*8
_output_shapes&
$:"                  X@*
T0
Т
!dropout3_mod/cond/Identity/SwitchSwitchpool3_mod/MaxPooldropout3_mod/cond/pred_id*
T0*$
_class
loc:@pool3_mod/MaxPool*\
_output_shapesJ
H:"                  X@:"                  X@
Е
dropout3_mod/cond/MergeMergedropout3_mod/cond/Identitydropout3_mod/cond/dropout/mul*
T0*
N*:
_output_shapes(
&:"                  X@: 
\
ShapeShapedropout3_mod/cond/Merge*
T0*
out_type0*
_output_shapes
:
]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
щ
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
_
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ђ
strided_slice_1StridedSliceShapestrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
W
flatten4_mod/shape/2Const*
value
B :ђ,*
dtype0*
_output_shapes
: 
і
flatten4_mod/shapePackstrided_slicestrided_slice_1flatten4_mod/shape/2*
T0*

axis *
N*
_output_shapes
:
њ
flatten4_modReshapedropout3_mod/cond/Mergeflatten4_mod/shape*
T0*
Tshape0*5
_output_shapes#
!:                  ђ,
Ц
0fc5_mod/weights/Initializer/random_uniform/shapeConst*
valueB"      *"
_class
loc:@fc5_mod/weights*
dtype0*
_output_shapes
:
Ќ
.fc5_mod/weights/Initializer/random_uniform/minConst*
valueB
 *з5й*"
_class
loc:@fc5_mod/weights*
dtype0*
_output_shapes
: 
Ќ
.fc5_mod/weights/Initializer/random_uniform/maxConst*
valueB
 *з5=*"
_class
loc:@fc5_mod/weights*
dtype0*
_output_shapes
: 
ы
8fc5_mod/weights/Initializer/random_uniform/RandomUniformRandomUniform0fc5_mod/weights/Initializer/random_uniform/shape*
T0*"
_class
loc:@fc5_mod/weights*
seed2А*
dtype0* 
_output_shapes
:
ђ,ђ*

seed*
┌
.fc5_mod/weights/Initializer/random_uniform/subSub.fc5_mod/weights/Initializer/random_uniform/max.fc5_mod/weights/Initializer/random_uniform/min*
T0*"
_class
loc:@fc5_mod/weights*
_output_shapes
: 
Ь
.fc5_mod/weights/Initializer/random_uniform/mulMul8fc5_mod/weights/Initializer/random_uniform/RandomUniform.fc5_mod/weights/Initializer/random_uniform/sub*
T0*"
_class
loc:@fc5_mod/weights* 
_output_shapes
:
ђ,ђ
Я
*fc5_mod/weights/Initializer/random_uniformAdd.fc5_mod/weights/Initializer/random_uniform/mul.fc5_mod/weights/Initializer/random_uniform/min*
T0*"
_class
loc:@fc5_mod/weights* 
_output_shapes
:
ђ,ђ
Ф
fc5_mod/weights
VariableV2*"
_class
loc:@fc5_mod/weights*
	container *
shape:
ђ,ђ*
dtype0* 
_output_shapes
:
ђ,ђ*
shared_name 
Н
fc5_mod/weights/AssignAssignfc5_mod/weights*fc5_mod/weights/Initializer/random_uniform*
T0*"
_class
loc:@fc5_mod/weights*
validate_shape(* 
_output_shapes
:
ђ,ђ*
use_locking(
ђ
fc5_mod/weights/readIdentityfc5_mod/weights*
T0*"
_class
loc:@fc5_mod/weights* 
_output_shapes
:
ђ,ђ
ъ
0fc5_mod/biases/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:ђ*!
_class
loc:@fc5_mod/biases
ј
&fc5_mod/biases/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@fc5_mod/biases*
dtype0*
_output_shapes
: 
П
 fc5_mod/biases/Initializer/zerosFill0fc5_mod/biases/Initializer/zeros/shape_as_tensor&fc5_mod/biases/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@fc5_mod/biases*
_output_shapes	
:ђ
Ъ
fc5_mod/biases
VariableV2*
shared_name *!
_class
loc:@fc5_mod/biases*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
├
fc5_mod/biases/AssignAssignfc5_mod/biases fc5_mod/biases/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@fc5_mod/biases*
validate_shape(*
_output_shapes	
:ђ
x
fc5_mod/biases/readIdentityfc5_mod/biases*
T0*!
_class
loc:@fc5_mod/biases*
_output_shapes	
:ђ
c
fc5_mod/Tensordot/ShapeShapeflatten4_mod*
T0*
out_type0*
_output_shapes
:
X
fc5_mod/Tensordot/RankConst*
value	B :*
dtype0*
_output_shapes
: 
`
fc5_mod/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
b
 fc5_mod/Tensordot/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
Ї
fc5_mod/Tensordot/GreaterEqualGreaterEqualfc5_mod/Tensordot/axes fc5_mod/Tensordot/GreaterEqual/y*
T0*
_output_shapes
:
r
fc5_mod/Tensordot/CastCastfc5_mod/Tensordot/GreaterEqual*
_output_shapes
:*

DstT0*

SrcT0

q
fc5_mod/Tensordot/mulMulfc5_mod/Tensordot/Castfc5_mod/Tensordot/axes*
T0*
_output_shapes
:
Z
fc5_mod/Tensordot/Less/yConst*
dtype0*
_output_shapes
: *
value	B : 
u
fc5_mod/Tensordot/LessLessfc5_mod/Tensordot/axesfc5_mod/Tensordot/Less/y*
T0*
_output_shapes
:
l
fc5_mod/Tensordot/Cast_1Castfc5_mod/Tensordot/Less*

SrcT0
*
_output_shapes
:*

DstT0
q
fc5_mod/Tensordot/addAddfc5_mod/Tensordot/axesfc5_mod/Tensordot/Rank*
T0*
_output_shapes
:
t
fc5_mod/Tensordot/mul_1Mulfc5_mod/Tensordot/Cast_1fc5_mod/Tensordot/add*
T0*
_output_shapes
:
s
fc5_mod/Tensordot/add_1Addfc5_mod/Tensordot/mulfc5_mod/Tensordot/mul_1*
T0*
_output_shapes
:
_
fc5_mod/Tensordot/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
fc5_mod/Tensordot/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ъ
fc5_mod/Tensordot/rangeRangefc5_mod/Tensordot/range/startfc5_mod/Tensordot/Rankfc5_mod/Tensordot/range/delta*
_output_shapes
:*

Tidx0
ц
fc5_mod/Tensordot/ListDiffListDifffc5_mod/Tensordot/rangefc5_mod/Tensordot/add_1*
T0*
out_idx0*2
_output_shapes 
:         :         
│
fc5_mod/Tensordot/GatherGatherfc5_mod/Tensordot/Shapefc5_mod/Tensordot/ListDiff*
Tparams0*
validate_indices(*#
_output_shapes
:         *
Tindices0
Е
fc5_mod/Tensordot/Gather_1Gatherfc5_mod/Tensordot/Shapefc5_mod/Tensordot/add_1*
Tindices0*
Tparams0*
validate_indices(*
_output_shapes
:
a
fc5_mod/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ј
fc5_mod/Tensordot/ProdProdfc5_mod/Tensordot/Gatherfc5_mod/Tensordot/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
c
fc5_mod/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ћ
fc5_mod/Tensordot/Prod_1Prodfc5_mod/Tensordot/Gather_1fc5_mod/Tensordot/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
_
fc5_mod/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╝
fc5_mod/Tensordot/concatConcatV2fc5_mod/Tensordot/Gather_1fc5_mod/Tensordot/Gatherfc5_mod/Tensordot/concat/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
a
fc5_mod/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
┐
fc5_mod/Tensordot/concat_1ConcatV2fc5_mod/Tensordot/ListDifffc5_mod/Tensordot/add_1fc5_mod/Tensordot/concat_1/axis*
N*#
_output_shapes
:         *

Tidx0*
T0
І
fc5_mod/Tensordot/stackPackfc5_mod/Tensordot/Prodfc5_mod/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
Д
fc5_mod/Tensordot/transpose	Transposeflatten4_modfc5_mod/Tensordot/concat_1*=
_output_shapes+
):'                           *
Tperm0*
T0
Б
fc5_mod/Tensordot/ReshapeReshapefc5_mod/Tensordot/transposefc5_mod/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:                  
s
"fc5_mod/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
ю
fc5_mod/Tensordot/transpose_1	Transposefc5_mod/weights/read"fc5_mod/Tensordot/transpose_1/perm*
T0* 
_output_shapes
:
ђ,ђ*
Tperm0
r
!fc5_mod/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
А
fc5_mod/Tensordot/Reshape_1Reshapefc5_mod/Tensordot/transpose_1!fc5_mod/Tensordot/Reshape_1/shape* 
_output_shapes
:
ђ,ђ*
T0*
Tshape0
│
fc5_mod/Tensordot/MatMulMatMulfc5_mod/Tensordot/Reshapefc5_mod/Tensordot/Reshape_1*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
d
fc5_mod/Tensordot/Const_2Const*
valueB:ђ*
dtype0*
_output_shapes
:
a
fc5_mod/Tensordot/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
┐
fc5_mod/Tensordot/concat_2ConcatV2fc5_mod/Tensordot/Gatherfc5_mod/Tensordot/Const_2fc5_mod/Tensordot/concat_2/axis*
N*#
_output_shapes
:         *

Tidx0*
T0
а
fc5_mod/TensordotReshapefc5_mod/Tensordot/MatMulfc5_mod/Tensordot/concat_2*
T0*
Tshape0*5
_output_shapes#
!:                  ђ
Ў
fc5_mod/BiasAddBiasAddfc5_mod/Tensordotfc5_mod/biases/read*
data_formatNHWC*5
_output_shapes#
!:                  ђ*
T0
e
fc5_mod/ReluRelufc5_mod/BiasAdd*5
_output_shapes#
!:                  ђ*
T0
c
dropout5_mod/cond/SwitchSwitchPlaceholder_2Placeholder_2*
T0
*
_output_shapes
: : 
c
dropout5_mod/cond/switch_tIdentitydropout5_mod/cond/Switch:1*
T0
*
_output_shapes
: 
a
dropout5_mod/cond/switch_fIdentitydropout5_mod/cond/Switch*
_output_shapes
: *
T0

U
dropout5_mod/cond/pred_idIdentityPlaceholder_2*
_output_shapes
: *
T0

Ё
#dropout5_mod/cond/dropout/keep_probConst^dropout5_mod/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
Є
dropout5_mod/cond/dropout/ShapeShape(dropout5_mod/cond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0
█
&dropout5_mod/cond/dropout/Shape/SwitchSwitchfc5_mod/Reludropout5_mod/cond/pred_id*
T0*
_class
loc:@fc5_mod/Relu*V
_output_shapesD
B:                  ђ:                  ђ
ј
,dropout5_mod/cond/dropout/random_uniform/minConst^dropout5_mod/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
ј
,dropout5_mod/cond/dropout/random_uniform/maxConst^dropout5_mod/cond/switch_t*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
¤
6dropout5_mod/cond/dropout/random_uniform/RandomUniformRandomUniformdropout5_mod/cond/dropout/Shape*
dtype0*5
_output_shapes#
!:                  ђ*
seed2Я*

seed**
T0
░
,dropout5_mod/cond/dropout/random_uniform/subSub,dropout5_mod/cond/dropout/random_uniform/max,dropout5_mod/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
┘
,dropout5_mod/cond/dropout/random_uniform/mulMul6dropout5_mod/cond/dropout/random_uniform/RandomUniform,dropout5_mod/cond/dropout/random_uniform/sub*
T0*5
_output_shapes#
!:                  ђ
╦
(dropout5_mod/cond/dropout/random_uniformAdd,dropout5_mod/cond/dropout/random_uniform/mul,dropout5_mod/cond/dropout/random_uniform/min*5
_output_shapes#
!:                  ђ*
T0
│
dropout5_mod/cond/dropout/addAdd#dropout5_mod/cond/dropout/keep_prob(dropout5_mod/cond/dropout/random_uniform*
T0*5
_output_shapes#
!:                  ђ
Є
dropout5_mod/cond/dropout/FloorFloordropout5_mod/cond/dropout/add*
T0*5
_output_shapes#
!:                  ђ
и
dropout5_mod/cond/dropout/divRealDiv(dropout5_mod/cond/dropout/Shape/Switch:1#dropout5_mod/cond/dropout/keep_prob*
T0*5
_output_shapes#
!:                  ђ
ц
dropout5_mod/cond/dropout/mulMuldropout5_mod/cond/dropout/divdropout5_mod/cond/dropout/Floor*5
_output_shapes#
!:                  ђ*
T0
Ѕ
dropout5_mod/cond/IdentityIdentity!dropout5_mod/cond/Identity/Switch*
T0*5
_output_shapes#
!:                  ђ
о
!dropout5_mod/cond/Identity/SwitchSwitchfc5_mod/Reludropout5_mod/cond/pred_id*V
_output_shapesD
B:                  ђ:                  ђ*
T0*
_class
loc:@fc5_mod/Relu
д
dropout5_mod/cond/MergeMergedropout5_mod/cond/Identitydropout5_mod/cond/dropout/mul*
T0*
N*7
_output_shapes%
#:                  ђ: 
Ц
0fc6_mod/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"   X   *"
_class
loc:@fc6_mod/weights
Ќ
.fc6_mod/weights/Initializer/random_uniform/minConst*
valueB
 *├лЙ*"
_class
loc:@fc6_mod/weights*
dtype0*
_output_shapes
: 
Ќ
.fc6_mod/weights/Initializer/random_uniform/maxConst*
valueB
 *├л>*"
_class
loc:@fc6_mod/weights*
dtype0*
_output_shapes
: 
­
8fc6_mod/weights/Initializer/random_uniform/RandomUniformRandomUniform0fc6_mod/weights/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	ђX*

seed**
T0*"
_class
loc:@fc6_mod/weights*
seed2Ь
┌
.fc6_mod/weights/Initializer/random_uniform/subSub.fc6_mod/weights/Initializer/random_uniform/max.fc6_mod/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@fc6_mod/weights
ь
.fc6_mod/weights/Initializer/random_uniform/mulMul8fc6_mod/weights/Initializer/random_uniform/RandomUniform.fc6_mod/weights/Initializer/random_uniform/sub*
_output_shapes
:	ђX*
T0*"
_class
loc:@fc6_mod/weights
▀
*fc6_mod/weights/Initializer/random_uniformAdd.fc6_mod/weights/Initializer/random_uniform/mul.fc6_mod/weights/Initializer/random_uniform/min*
T0*"
_class
loc:@fc6_mod/weights*
_output_shapes
:	ђX
Е
fc6_mod/weights
VariableV2*
dtype0*
_output_shapes
:	ђX*
shared_name *"
_class
loc:@fc6_mod/weights*
	container *
shape:	ђX
н
fc6_mod/weights/AssignAssignfc6_mod/weights*fc6_mod/weights/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@fc6_mod/weights*
validate_shape(*
_output_shapes
:	ђX

fc6_mod/weights/readIdentityfc6_mod/weights*
_output_shapes
:	ђX*
T0*"
_class
loc:@fc6_mod/weights
Ю
0fc6_mod/biases/Initializer/zeros/shape_as_tensorConst*
valueB:X*!
_class
loc:@fc6_mod/biases*
dtype0*
_output_shapes
:
ј
&fc6_mod/biases/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@fc6_mod/biases*
dtype0*
_output_shapes
: 
▄
 fc6_mod/biases/Initializer/zerosFill0fc6_mod/biases/Initializer/zeros/shape_as_tensor&fc6_mod/biases/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@fc6_mod/biases*
_output_shapes
:X
Ю
fc6_mod/biases
VariableV2*!
_class
loc:@fc6_mod/biases*
	container *
shape:X*
dtype0*
_output_shapes
:X*
shared_name 
┬
fc6_mod/biases/AssignAssignfc6_mod/biases fc6_mod/biases/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@fc6_mod/biases*
validate_shape(*
_output_shapes
:X
w
fc6_mod/biases/readIdentityfc6_mod/biases*
_output_shapes
:X*
T0*!
_class
loc:@fc6_mod/biases
n
fc6_mod/Tensordot/ShapeShapedropout5_mod/cond/Merge*
_output_shapes
:*
T0*
out_type0
X
fc6_mod/Tensordot/RankConst*
dtype0*
_output_shapes
: *
value	B :
`
fc6_mod/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
b
 fc6_mod/Tensordot/GreaterEqual/yConst*
dtype0*
_output_shapes
: *
value	B : 
Ї
fc6_mod/Tensordot/GreaterEqualGreaterEqualfc6_mod/Tensordot/axes fc6_mod/Tensordot/GreaterEqual/y*
_output_shapes
:*
T0
r
fc6_mod/Tensordot/CastCastfc6_mod/Tensordot/GreaterEqual*

SrcT0
*
_output_shapes
:*

DstT0
q
fc6_mod/Tensordot/mulMulfc6_mod/Tensordot/Castfc6_mod/Tensordot/axes*
T0*
_output_shapes
:
Z
fc6_mod/Tensordot/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
u
fc6_mod/Tensordot/LessLessfc6_mod/Tensordot/axesfc6_mod/Tensordot/Less/y*
T0*
_output_shapes
:
l
fc6_mod/Tensordot/Cast_1Castfc6_mod/Tensordot/Less*

SrcT0
*
_output_shapes
:*

DstT0
q
fc6_mod/Tensordot/addAddfc6_mod/Tensordot/axesfc6_mod/Tensordot/Rank*
T0*
_output_shapes
:
t
fc6_mod/Tensordot/mul_1Mulfc6_mod/Tensordot/Cast_1fc6_mod/Tensordot/add*
_output_shapes
:*
T0
s
fc6_mod/Tensordot/add_1Addfc6_mod/Tensordot/mulfc6_mod/Tensordot/mul_1*
T0*
_output_shapes
:
_
fc6_mod/Tensordot/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
fc6_mod/Tensordot/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ъ
fc6_mod/Tensordot/rangeRangefc6_mod/Tensordot/range/startfc6_mod/Tensordot/Rankfc6_mod/Tensordot/range/delta*
_output_shapes
:*

Tidx0
ц
fc6_mod/Tensordot/ListDiffListDifffc6_mod/Tensordot/rangefc6_mod/Tensordot/add_1*
T0*
out_idx0*2
_output_shapes 
:         :         
│
fc6_mod/Tensordot/GatherGatherfc6_mod/Tensordot/Shapefc6_mod/Tensordot/ListDiff*#
_output_shapes
:         *
Tindices0*
Tparams0*
validate_indices(
Е
fc6_mod/Tensordot/Gather_1Gatherfc6_mod/Tensordot/Shapefc6_mod/Tensordot/add_1*
Tindices0*
Tparams0*
validate_indices(*
_output_shapes
:
a
fc6_mod/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ј
fc6_mod/Tensordot/ProdProdfc6_mod/Tensordot/Gatherfc6_mod/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
c
fc6_mod/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Ћ
fc6_mod/Tensordot/Prod_1Prodfc6_mod/Tensordot/Gather_1fc6_mod/Tensordot/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
_
fc6_mod/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
╝
fc6_mod/Tensordot/concatConcatV2fc6_mod/Tensordot/Gather_1fc6_mod/Tensordot/Gatherfc6_mod/Tensordot/concat/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
a
fc6_mod/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
┐
fc6_mod/Tensordot/concat_1ConcatV2fc6_mod/Tensordot/ListDifffc6_mod/Tensordot/add_1fc6_mod/Tensordot/concat_1/axis*
N*#
_output_shapes
:         *

Tidx0*
T0
І
fc6_mod/Tensordot/stackPackfc6_mod/Tensordot/Prodfc6_mod/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
▓
fc6_mod/Tensordot/transpose	Transposedropout5_mod/cond/Mergefc6_mod/Tensordot/concat_1*
Tperm0*
T0*=
_output_shapes+
):'                           
Б
fc6_mod/Tensordot/ReshapeReshapefc6_mod/Tensordot/transposefc6_mod/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:                  
s
"fc6_mod/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
Џ
fc6_mod/Tensordot/transpose_1	Transposefc6_mod/weights/read"fc6_mod/Tensordot/transpose_1/perm*
T0*
_output_shapes
:	ђX*
Tperm0
r
!fc6_mod/Tensordot/Reshape_1/shapeConst*
valueB"   X   *
dtype0*
_output_shapes
:
а
fc6_mod/Tensordot/Reshape_1Reshapefc6_mod/Tensordot/transpose_1!fc6_mod/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	ђX
▓
fc6_mod/Tensordot/MatMulMatMulfc6_mod/Tensordot/Reshapefc6_mod/Tensordot/Reshape_1*
T0*'
_output_shapes
:         X*
transpose_a( *
transpose_b( 
c
fc6_mod/Tensordot/Const_2Const*
valueB:X*
dtype0*
_output_shapes
:
a
fc6_mod/Tensordot/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
┐
fc6_mod/Tensordot/concat_2ConcatV2fc6_mod/Tensordot/Gatherfc6_mod/Tensordot/Const_2fc6_mod/Tensordot/concat_2/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
Ъ
fc6_mod/TensordotReshapefc6_mod/Tensordot/MatMulfc6_mod/Tensordot/concat_2*
T0*
Tshape0*4
_output_shapes"
 :                  X
ў
fc6_mod/BiasAddBiasAddfc6_mod/Tensordotfc6_mod/biases/read*
T0*
data_formatNHWC*4
_output_shapes"
 :                  X
j
fc6_mod/SigmoidSigmoidfc6_mod/BiasAdd*
T0*4
_output_shapes"
 :                  X
S
log_loss/add/yConst*
valueB
 *Ћ┐о3*
dtype0*
_output_shapes
: 
s
log_loss/addAddfc6_mod/Sigmoidlog_loss/add/y*
T0*4
_output_shapes"
 :                  X
`
log_loss/LogLoglog_loss/add*
T0*4
_output_shapes"
 :                  X
m
log_loss/MulMulPlaceholderlog_loss/Log*4
_output_shapes"
 :                  X*
T0
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
log_loss/subSublog_loss/sub/xPlaceholder*
T0*4
_output_shapes"
 :                  X
U
log_loss/sub_1/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
w
log_loss/sub_1Sublog_loss/sub_1/xfc6_mod/Sigmoid*
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
log_loss/add_1Addlog_loss/sub_1log_loss/add_1/y*
T0*4
_output_shapes"
 :                  X
d
log_loss/Log_1Loglog_loss/add_1*
T0*4
_output_shapes"
 :                  X
r
log_loss/Mul_1Mullog_loss/sublog_loss/Log_1*
T0*4
_output_shapes"
 :                  X
r
log_loss/sub_2Sublog_loss/Neglog_loss/Mul_1*
T0*4
_output_shapes"
 :                  X
x
+log_loss/assert_broadcastable/weights/shapeShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
l
*log_loss/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B :
x
*log_loss/assert_broadcastable/values/shapeShapelog_loss/sub_2*
T0*
out_type0*
_output_shapes
:
k
)log_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
k
)log_loss/assert_broadcastable/is_scalar/xConst*
dtype0*
_output_shapes
: *
value	B : 
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
5log_loss/assert_broadcastable/is_valid_shape/switch_tIdentity5log_loss/assert_broadcastable/is_valid_shape/Switch:1*
_output_shapes
: *
T0

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
5log_loss/assert_broadcastable/is_valid_shape/Switch_1Switch'log_loss/assert_broadcastable/is_scalar4log_loss/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0
*:
_class0
.,loc:@log_loss/assert_broadcastable/is_scalar
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
\log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch*log_loss/assert_broadcastable/weights/rank4log_loss/assert_broadcastable/is_valid_shape/pred_id*
T0*=
_class3
1/loc:@log_loss/assert_broadcastable/weights/rank*
_output_shapes
: : 
ц
Mlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchSlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankSlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
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
Nlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentitySlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: *
T0

Ѓ
flog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
Ч
blog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsmlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1flog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*
_output_shapes

:*

Tdim0
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
glog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstP^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
Ш
alog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillglog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shapeglog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
_output_shapes

:*
T0*

index_type0
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
plog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationdlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1^log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*<
_output_shapes*
(:         :         :*
set_operationa-b*
T0*
validate_indices(
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
Wlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualYlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xhlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
Ѕ
Olog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1SwitchSlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankNlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*f
_class\
ZXloc:@log_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
Ф
Llog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeOlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Wlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
Ь
2log_loss/assert_broadcastable/is_valid_shape/MergeMergeLlog_loss/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge7log_loss/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
І
#log_loss/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
t
%log_loss/assert_broadcastable/Const_1Const*
dtype0*
_output_shapes
: *
valueB Bweights.shape=
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
2log_loss/assert_broadcastable/AssertGuard/switch_fIdentity0log_loss/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
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
valueB BPlaceholder_1:0*
dtype0*
_output_shapes
: 
║
7log_loss/assert_broadcastable/AssertGuard/Assert/data_4Const3^log_loss/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
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
7log_loss/assert_broadcastable/AssertGuard/Assert/SwitchSwitch2log_loss/assert_broadcastable/is_valid_shape/Merge1log_loss/assert_broadcastable/AssertGuard/pred_id*
T0
*E
_class;
97loc:@log_loss/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
ј
9log_loss/assert_broadcastable/AssertGuard/Assert/Switch_1Switch+log_loss/assert_broadcastable/weights/shape1log_loss/assert_broadcastable/AssertGuard/pred_id* 
_output_shapes
::*
T0*>
_class4
20loc:@log_loss/assert_broadcastable/weights/shape
ї
9log_loss/assert_broadcastable/AssertGuard/Assert/Switch_2Switch*log_loss/assert_broadcastable/values/shape1log_loss/assert_broadcastable/AssertGuard/pred_id*
T0*=
_class3
1/loc:@log_loss/assert_broadcastable/values/shape* 
_output_shapes
::
■
9log_loss/assert_broadcastable/AssertGuard/Assert/Switch_3Switch'log_loss/assert_broadcastable/is_scalar1log_loss/assert_broadcastable/AssertGuard/pred_id*
_output_shapes
: : *
T0
*:
_class0
.,loc:@log_loss/assert_broadcastable/is_scalar
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
log_loss/ConstConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
:*!
valueB"          
q
log_loss/SumSumlog_loss/Mul_2log_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Њ
log_loss/num_present/Equal/yConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB
 *    
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
log_loss/num_present/SelectSelectlog_loss/num_present/Equallog_loss/num_present/zeros_likelog_loss/num_present/ones_like*
T0*4
_output_shapes"
 :                  X
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
Glog_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
value	B :*
dtype0*
_output_shapes
: 
╗
Glog_loss/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
value	B : 
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
Slog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentityQlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
к
Rlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityElog_loss/num_present/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: *
T0

ш
Slog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchElog_loss/num_present/broadcast_weights/assert_broadcastable/is_scalarRlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0
*X
_classN
LJloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
Љ
qlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualxlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchzlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
ъ
xlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchGlog_loss/num_present/broadcast_weights/assert_broadcastable/values/rankRlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*Z
_classP
NLloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/values/rank*
_output_shapes
: : 
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
mlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitymlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
Є
mlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityklog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
ї
llog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityqlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: *
T0

Ы
ёlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst0^log_loss/assert_broadcastable/AssertGuard/Mergen^log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
valueB :
         
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
log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillЁlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeЁlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*

index_type0*
_output_shapes

:
Т
Ђlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst0^log_loss/assert_broadcastable/AssertGuard/Mergen^log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╬
|log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2ђlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimslog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeЂlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
N*
_output_shapes

:*

Tidx0*
T0
З
єlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst0^log_loss/assert_broadcastable/AssertGuard/Mergen^log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
valueB :
         
▀
ѓlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsЇlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1єlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

:
╝
Ѕlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchIlog_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeRlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*\
_classR
PNloc:@log_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
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
єlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeљlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
_output_shapes
: *
T0*
out_type0
█
wlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst0^log_loss/assert_broadcastable/AssertGuard/Mergen^log_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
А
ulog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualwlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xєlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
_output_shapes
: *
T0
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
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
М
Clog_loss/num_present/broadcast_weights/assert_broadcastable/Const_2Const0^log_loss/assert_broadcastable/AssertGuard/Merge*.
value%B# Blog_loss/num_present/Select:0*
dtype0*
_output_shapes
: 
├
Clog_loss/num_present/broadcast_weights/assert_broadcastable/Const_3Const0^log_loss/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB Bvalues.shape=
к
Clog_loss/num_present/broadcast_weights/assert_broadcastable/Const_4Const0^log_loss/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *!
valueB Blog_loss/sub_2:0
└
Clog_loss/num_present/broadcast_weights/assert_broadcastable/Const_5Const0^log_loss/assert_broadcastable/AssertGuard/Merge*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Ъ
Nlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchPlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergePlog_loss/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
¤
Plog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityPlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
═
Plog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityNlog_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
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
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
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
6log_loss/num_present/broadcast_weights/ones_like/ConstConst0^log_loss/assert_broadcastable/AssertGuard/MergeN^log_loss/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
щ
0log_loss/num_present/broadcast_weights/ones_likeFill6log_loss/num_present/broadcast_weights/ones_like/Shape6log_loss/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*4
_output_shapes"
 :                  X
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
log_loss/num_presentSum&log_loss/num_present/broadcast_weightslog_loss/num_present/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
log_loss/Greater/yConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
valueB
 *    *
dtype0*
_output_shapes
: 
f
log_loss/GreaterGreaterlog_loss/num_presentlog_loss/Greater/y*
T0*
_output_shapes
: 
Є
log_loss/Equal/yConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
valueB
 *    *
dtype0*
_output_shapes
: 
`
log_loss/EqualEquallog_loss/num_presentlog_loss/Equal/y*
T0*
_output_shapes
: 
Ї
log_loss/ones_like/ShapeConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
valueB *
dtype0*
_output_shapes
: 
Ј
log_loss/ones_like/ConstConst0^log_loss/assert_broadcastable/AssertGuard/Merge*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Ђ
log_loss/ones_likeFilllog_loss/ones_like/Shapelog_loss/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
t
log_loss/SelectSelectlog_loss/Equallog_loss/ones_likelog_loss/num_present*
T0*
_output_shapes
: 
Y
log_loss/divRealDivlog_loss/Sum_1log_loss/Select*
T0*
_output_shapes
: 
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
log_loss/valueSelectlog_loss/Greaterlog_loss/divlog_loss/zeros_like*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
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
.gradients/log_loss/value_grad/zeros_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
&gradients/log_loss/value_grad/Select_1Selectlog_loss/Greater(gradients/log_loss/value_grad/zeros_likegradients/Fill*
T0*
_output_shapes
: 
є
.gradients/log_loss/value_grad/tuple/group_depsNoOp%^gradients/log_loss/value_grad/Select'^gradients/log_loss/value_grad/Select_1
з
6gradients/log_loss/value_grad/tuple/control_dependencyIdentity$gradients/log_loss/value_grad/Select/^gradients/log_loss/value_grad/tuple/group_deps*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients/log_loss/value_grad/Select
щ
8gradients/log_loss/value_grad/tuple/control_dependency_1Identity&gradients/log_loss/value_grad/Select_1/^gradients/log_loss/value_grad/tuple/group_deps*
_output_shapes
: *
T0*9
_class/
-+loc:@gradients/log_loss/value_grad/Select_1
d
!gradients/log_loss/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
f
#gradients/log_loss/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
¤
1gradients/log_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/log_loss/div_grad/Shape#gradients/log_loss/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ў
#gradients/log_loss/div_grad/RealDivRealDiv6gradients/log_loss/value_grad/tuple/control_dependencylog_loss/Select*
T0*
_output_shapes
: 
Й
gradients/log_loss/div_grad/SumSum#gradients/log_loss/div_grad/RealDiv1gradients/log_loss/div_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
А
#gradients/log_loss/div_grad/ReshapeReshapegradients/log_loss/div_grad/Sum!gradients/log_loss/div_grad/Shape*
_output_shapes
: *
T0*
Tshape0
W
gradients/log_loss/div_grad/NegNeglog_loss/Sum_1*
T0*
_output_shapes
: 
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
gradients/log_loss/div_grad/mulMul6gradients/log_loss/value_grad/tuple/control_dependency%gradients/log_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0
Й
!gradients/log_loss/div_grad/Sum_1Sumgradients/log_loss/div_grad/mul3gradients/log_loss/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Д
%gradients/log_loss/div_grad/Reshape_1Reshape!gradients/log_loss/div_grad/Sum_1#gradients/log_loss/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ѓ
,gradients/log_loss/div_grad/tuple/group_depsNoOp$^gradients/log_loss/div_grad/Reshape&^gradients/log_loss/div_grad/Reshape_1
ь
4gradients/log_loss/div_grad/tuple/control_dependencyIdentity#gradients/log_loss/div_grad/Reshape-^gradients/log_loss/div_grad/tuple/group_deps*
_output_shapes
: *
T0*6
_class,
*(loc:@gradients/log_loss/div_grad/Reshape
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
"gradients/log_loss/Sum_1_grad/TileTile%gradients/log_loss/Sum_1_grad/Reshape,gradients/log_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
: 
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
'gradients/log_loss/Select_grad/Select_1Selectlog_loss/Equal)gradients/log_loss/Select_grad/zeros_like6gradients/log_loss/div_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
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
 gradients/log_loss/Sum_grad/TileTile#gradients/log_loss/Sum_grad/Reshape!gradients/log_loss/Sum_grad/Shape*
T0*4
_output_shapes"
 :                  X*

Tmultiples0
q
#gradients/log_loss/Mul_2_grad/ShapeShapelog_loss/sub_2*
T0*
out_type0*
_output_shapes
:
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
!gradients/log_loss/Mul_2_grad/mulMul gradients/log_loss/Sum_grad/TilePlaceholder_1*4
_output_shapes"
 :                  X*
T0
└
!gradients/log_loss/Mul_2_grad/SumSum!gradients/log_loss/Mul_2_grad/mul3gradients/log_loss/Mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
┼
%gradients/log_loss/Mul_2_grad/ReshapeReshape!gradients/log_loss/Mul_2_grad/Sum#gradients/log_loss/Mul_2_grad/Shape*4
_output_shapes"
 :                  X*
T0*
Tshape0
Џ
#gradients/log_loss/Mul_2_grad/mul_1Mullog_loss/sub_2 gradients/log_loss/Sum_grad/Tile*
T0*4
_output_shapes"
 :                  X
к
#gradients/log_loss/Mul_2_grad/Sum_1Sum#gradients/log_loss/Mul_2_grad/mul_15gradients/log_loss/Mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
╦
'gradients/log_loss/Mul_2_grad/Reshape_1Reshape#gradients/log_loss/Mul_2_grad/Sum_1%gradients/log_loss/Mul_2_grad/Shape_1*4
_output_shapes"
 :                  X*
T0*
Tshape0
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
+gradients/log_loss/num_present_grad/ReshapeReshape9gradients/log_loss/Select_grad/tuple/control_dependency_11gradients/log_loss/num_present_grad/Reshape/shape*"
_output_shapes
:*
T0*
Tshape0
Ј
)gradients/log_loss/num_present_grad/ShapeShape&log_loss/num_present/broadcast_weights*
T0*
out_type0*
_output_shapes
:
┘
(gradients/log_loss/num_present_grad/TileTile+gradients/log_loss/num_present_grad/Reshape)gradients/log_loss/num_present_grad/Shape*4
_output_shapes"
 :                  X*

Tmultiples0*
T0
ќ
;gradients/log_loss/num_present/broadcast_weights_grad/ShapeShapelog_loss/num_present/Select*
T0*
out_type0*
_output_shapes
:
Г
=gradients/log_loss/num_present/broadcast_weights_grad/Shape_1Shape0log_loss/num_present/broadcast_weights/ones_like*
T0*
out_type0*
_output_shapes
:
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
9gradients/log_loss/num_present/broadcast_weights_grad/SumSum9gradients/log_loss/num_present/broadcast_weights_grad/mulKgradients/log_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
;gradients/log_loss/num_present/broadcast_weights_grad/Sum_1Sum;gradients/log_loss/num_present/broadcast_weights_grad/mul_1Mgradients/log_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Њ
?gradients/log_loss/num_present/broadcast_weights_grad/Reshape_1Reshape;gradients/log_loss/num_present/broadcast_weights_grad/Sum_1=gradients/log_loss/num_present/broadcast_weights_grad/Shape_1*4
_output_shapes"
 :                  X*
T0*
Tshape0
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
Pgradients/log_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Identity?gradients/log_loss/num_present/broadcast_weights_grad/Reshape_1G^gradients/log_loss/num_present/broadcast_weights_grad/tuple/group_deps*4
_output_shapes"
 :                  X*
T0*R
_classH
FDloc:@gradients/log_loss/num_present/broadcast_weights_grad/Reshape_1
џ
Egradients/log_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*!
valueB"          *
dtype0*
_output_shapes
:
А
Cgradients/log_loss/num_present/broadcast_weights/ones_like_grad/SumSumPgradients/log_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Egradients/log_loss/num_present/broadcast_weights/ones_like_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
o
#gradients/log_loss/sub_2_grad/ShapeShapelog_loss/Neg*
T0*
out_type0*
_output_shapes
:
s
%gradients/log_loss/sub_2_grad/Shape_1Shapelog_loss/Mul_1*
_output_shapes
:*
T0*
out_type0
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
%gradients/log_loss/sub_2_grad/ReshapeReshape!gradients/log_loss/sub_2_grad/Sum#gradients/log_loss/sub_2_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :                  X
┘
#gradients/log_loss/sub_2_grad/Sum_1Sum6gradients/log_loss/Mul_2_grad/tuple/control_dependency5gradients/log_loss/sub_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
p
!gradients/log_loss/sub_2_grad/NegNeg#gradients/log_loss/sub_2_grad/Sum_1*
_output_shapes
:*
T0
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
gradients/log_loss/Neg_grad/NegNeg6gradients/log_loss/sub_2_grad/tuple/control_dependency*
T0*4
_output_shapes"
 :                  X
o
#gradients/log_loss/Mul_1_grad/ShapeShapelog_loss/sub*
T0*
out_type0*
_output_shapes
:
s
%gradients/log_loss/Mul_1_grad/Shape_1Shapelog_loss/Log_1*
_output_shapes
:*
T0*
out_type0
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
!gradients/log_loss/Mul_1_grad/SumSum!gradients/log_loss/Mul_1_grad/mul3gradients/log_loss/Mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
#gradients/log_loss/Mul_1_grad/Sum_1Sum#gradients/log_loss/Mul_1_grad/mul_15gradients/log_loss/Mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
╦
'gradients/log_loss/Mul_1_grad/Reshape_1Reshape#gradients/log_loss/Mul_1_grad/Sum_1%gradients/log_loss/Mul_1_grad/Shape_1*
T0*
Tshape0*4
_output_shapes"
 :                  X
ѕ
.gradients/log_loss/Mul_1_grad/tuple/group_depsNoOp&^gradients/log_loss/Mul_1_grad/Reshape(^gradients/log_loss/Mul_1_grad/Reshape_1
Њ
6gradients/log_loss/Mul_1_grad/tuple/control_dependencyIdentity%gradients/log_loss/Mul_1_grad/Reshape/^gradients/log_loss/Mul_1_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/Mul_1_grad/Reshape*4
_output_shapes"
 :                  X
Ў
8gradients/log_loss/Mul_1_grad/tuple/control_dependency_1Identity'gradients/log_loss/Mul_1_grad/Reshape_1/^gradients/log_loss/Mul_1_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/log_loss/Mul_1_grad/Reshape_1*4
_output_shapes"
 :                  X
l
!gradients/log_loss/Mul_grad/ShapeShapePlaceholder*
T0*
out_type0*
_output_shapes
:
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
gradients/log_loss/Mul_grad/mulMulgradients/log_loss/Neg_grad/Neglog_loss/Log*
T0*4
_output_shapes"
 :                  X
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
!gradients/log_loss/Mul_grad/Sum_1Sum!gradients/log_loss/Mul_grad/mul_13gradients/log_loss/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
┼
%gradients/log_loss/Mul_grad/Reshape_1Reshape!gradients/log_loss/Mul_grad/Sum_1#gradients/log_loss/Mul_grad/Shape_1*4
_output_shapes"
 :                  X*
T0*
Tshape0
ѓ
,gradients/log_loss/Mul_grad/tuple/group_depsNoOp$^gradients/log_loss/Mul_grad/Reshape&^gradients/log_loss/Mul_grad/Reshape_1
І
4gradients/log_loss/Mul_grad/tuple/control_dependencyIdentity#gradients/log_loss/Mul_grad/Reshape-^gradients/log_loss/Mul_grad/tuple/group_deps*4
_output_shapes"
 :                  X*
T0*6
_class,
*(loc:@gradients/log_loss/Mul_grad/Reshape
Љ
6gradients/log_loss/Mul_grad/tuple/control_dependency_1Identity%gradients/log_loss/Mul_grad/Reshape_1-^gradients/log_loss/Mul_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/Mul_grad/Reshape_1*4
_output_shapes"
 :                  X
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
gradients/log_loss/Log_grad/mulMul6gradients/log_loss/Mul_grad/tuple/control_dependency_1&gradients/log_loss/Log_grad/Reciprocal*
T0*4
_output_shapes"
 :                  X
q
#gradients/log_loss/add_1_grad/ShapeShapelog_loss/sub_1*
_output_shapes
:*
T0*
out_type0
h
%gradients/log_loss/add_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Н
3gradients/log_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/log_loss/add_1_grad/Shape%gradients/log_loss/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
└
!gradients/log_loss/add_1_grad/SumSum!gradients/log_loss/Log_1_grad/mul3gradients/log_loss/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
┼
%gradients/log_loss/add_1_grad/ReshapeReshape!gradients/log_loss/add_1_grad/Sum#gradients/log_loss/add_1_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :                  X
─
#gradients/log_loss/add_1_grad/Sum_1Sum!gradients/log_loss/Log_1_grad/mul5gradients/log_loss/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Г
'gradients/log_loss/add_1_grad/Reshape_1Reshape#gradients/log_loss/add_1_grad/Sum_1%gradients/log_loss/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ѕ
.gradients/log_loss/add_1_grad/tuple/group_depsNoOp&^gradients/log_loss/add_1_grad/Reshape(^gradients/log_loss/add_1_grad/Reshape_1
Њ
6gradients/log_loss/add_1_grad/tuple/control_dependencyIdentity%gradients/log_loss/add_1_grad/Reshape/^gradients/log_loss/add_1_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/add_1_grad/Reshape*4
_output_shapes"
 :                  X
ч
8gradients/log_loss/add_1_grad/tuple/control_dependency_1Identity'gradients/log_loss/add_1_grad/Reshape_1/^gradients/log_loss/add_1_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/log_loss/add_1_grad/Reshape_1*
_output_shapes
: 
p
!gradients/log_loss/add_grad/ShapeShapefc6_mod/Sigmoid*
_output_shapes
:*
T0*
out_type0
f
#gradients/log_loss/add_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
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
4gradients/log_loss/add_grad/tuple/control_dependencyIdentity#gradients/log_loss/add_grad/Reshape-^gradients/log_loss/add_grad/tuple/group_deps*4
_output_shapes"
 :                  X*
T0*6
_class,
*(loc:@gradients/log_loss/add_grad/Reshape
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
t
%gradients/log_loss/sub_1_grad/Shape_1Shapefc6_mod/Sigmoid*
T0*
out_type0*
_output_shapes
:
Н
3gradients/log_loss/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/log_loss/sub_1_grad/Shape%gradients/log_loss/sub_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Н
!gradients/log_loss/sub_1_grad/SumSum6gradients/log_loss/add_1_grad/tuple/control_dependency3gradients/log_loss/sub_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
6gradients/log_loss/sub_1_grad/tuple/control_dependencyIdentity%gradients/log_loss/sub_1_grad/Reshape/^gradients/log_loss/sub_1_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/log_loss/sub_1_grad/Reshape*
_output_shapes
: 
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
Ў
*gradients/fc6_mod/Sigmoid_grad/SigmoidGradSigmoidGradfc6_mod/Sigmoidgradients/AddN*
T0*4
_output_shapes"
 :                  X
А
*gradients/fc6_mod/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/fc6_mod/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:X
Љ
/gradients/fc6_mod/BiasAdd_grad/tuple/group_depsNoOp+^gradients/fc6_mod/Sigmoid_grad/SigmoidGrad+^gradients/fc6_mod/BiasAdd_grad/BiasAddGrad
Ъ
7gradients/fc6_mod/BiasAdd_grad/tuple/control_dependencyIdentity*gradients/fc6_mod/Sigmoid_grad/SigmoidGrad0^gradients/fc6_mod/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/fc6_mod/Sigmoid_grad/SigmoidGrad*4
_output_shapes"
 :                  X
Є
9gradients/fc6_mod/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/fc6_mod/BiasAdd_grad/BiasAddGrad0^gradients/fc6_mod/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/fc6_mod/BiasAdd_grad/BiasAddGrad*
_output_shapes
:X
~
&gradients/fc6_mod/Tensordot_grad/ShapeShapefc6_mod/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
н
(gradients/fc6_mod/Tensordot_grad/ReshapeReshape7gradients/fc6_mod/BiasAdd_grad/tuple/control_dependency&gradients/fc6_mod/Tensordot_grad/Shape*'
_output_shapes
:         X*
T0*
Tshape0
п
.gradients/fc6_mod/Tensordot/MatMul_grad/MatMulMatMul(gradients/fc6_mod/Tensordot_grad/Reshapefc6_mod/Tensordot/Reshape_1*
transpose_b(*
T0*(
_output_shapes
:         ђ*
transpose_a( 
О
0gradients/fc6_mod/Tensordot/MatMul_grad/MatMul_1MatMulfc6_mod/Tensordot/Reshape(gradients/fc6_mod/Tensordot_grad/Reshape*'
_output_shapes
:         X*
transpose_a(*
transpose_b( *
T0
ц
8gradients/fc6_mod/Tensordot/MatMul_grad/tuple/group_depsNoOp/^gradients/fc6_mod/Tensordot/MatMul_grad/MatMul1^gradients/fc6_mod/Tensordot/MatMul_grad/MatMul_1
Г
@gradients/fc6_mod/Tensordot/MatMul_grad/tuple/control_dependencyIdentity.gradients/fc6_mod/Tensordot/MatMul_grad/MatMul9^gradients/fc6_mod/Tensordot/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*A
_class7
53loc:@gradients/fc6_mod/Tensordot/MatMul_grad/MatMul
ф
Bgradients/fc6_mod/Tensordot/MatMul_grad/tuple/control_dependency_1Identity0gradients/fc6_mod/Tensordot/MatMul_grad/MatMul_19^gradients/fc6_mod/Tensordot/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/fc6_mod/Tensordot/MatMul_grad/MatMul_1*
_output_shapes
:	ђX
Ѕ
.gradients/fc6_mod/Tensordot/Reshape_grad/ShapeShapefc6_mod/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
Ѓ
0gradients/fc6_mod/Tensordot/Reshape_grad/ReshapeReshape@gradients/fc6_mod/Tensordot/MatMul_grad/tuple/control_dependency.gradients/fc6_mod/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*=
_output_shapes+
):'                           
Ђ
0gradients/fc6_mod/Tensordot/Reshape_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"   X   
в
2gradients/fc6_mod/Tensordot/Reshape_1_grad/ReshapeReshapeBgradients/fc6_mod/Tensordot/MatMul_grad/tuple/control_dependency_10gradients/fc6_mod/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:	ђX
Џ
<gradients/fc6_mod/Tensordot/transpose_grad/InvertPermutationInvertPermutationfc6_mod/Tensordot/concat_1*
T0*#
_output_shapes
:         
■
4gradients/fc6_mod/Tensordot/transpose_grad/transpose	Transpose0gradients/fc6_mod/Tensordot/Reshape_grad/Reshape<gradients/fc6_mod/Tensordot/transpose_grad/InvertPermutation*
Tperm0*
T0*5
_output_shapes#
!:                  ђ
ю
>gradients/fc6_mod/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation"fc6_mod/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
Ь
6gradients/fc6_mod/Tensordot/transpose_1_grad/transpose	Transpose2gradients/fc6_mod/Tensordot/Reshape_1_grad/Reshape>gradients/fc6_mod/Tensordot/transpose_1_grad/InvertPermutation*
T0*
_output_shapes
:	ђX*
Tperm0
х
0gradients/dropout5_mod/cond/Merge_grad/cond_gradSwitch4gradients/fc6_mod/Tensordot/transpose_grad/transposedropout5_mod/cond/pred_id*V
_output_shapesD
B:                  ђ:                  ђ*
T0*G
_class=
;9loc:@gradients/fc6_mod/Tensordot/transpose_grad/transpose
r
7gradients/dropout5_mod/cond/Merge_grad/tuple/group_depsNoOp1^gradients/dropout5_mod/cond/Merge_grad/cond_grad
└
?gradients/dropout5_mod/cond/Merge_grad/tuple/control_dependencyIdentity0gradients/dropout5_mod/cond/Merge_grad/cond_grad8^gradients/dropout5_mod/cond/Merge_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/fc6_mod/Tensordot/transpose_grad/transpose*5
_output_shapes#
!:                  ђ
─
Agradients/dropout5_mod/cond/Merge_grad/tuple/control_dependency_1Identity2gradients/dropout5_mod/cond/Merge_grad/cond_grad:18^gradients/dropout5_mod/cond/Merge_grad/tuple/group_deps*5
_output_shapes#
!:                  ђ*
T0*G
_class=
;9loc:@gradients/fc6_mod/Tensordot/transpose_grad/transpose
Ј
2gradients/dropout5_mod/cond/dropout/mul_grad/ShapeShapedropout5_mod/cond/dropout/div*
_output_shapes
:*
T0*
out_type0
Њ
4gradients/dropout5_mod/cond/dropout/mul_grad/Shape_1Shapedropout5_mod/cond/dropout/Floor*
T0*
out_type0*
_output_shapes
:
ѓ
Bgradients/dropout5_mod/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/dropout5_mod/cond/dropout/mul_grad/Shape4gradients/dropout5_mod/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
█
0gradients/dropout5_mod/cond/dropout/mul_grad/mulMulAgradients/dropout5_mod/cond/Merge_grad/tuple/control_dependency_1dropout5_mod/cond/dropout/Floor*
T0*5
_output_shapes#
!:                  ђ
ь
0gradients/dropout5_mod/cond/dropout/mul_grad/SumSum0gradients/dropout5_mod/cond/dropout/mul_grad/mulBgradients/dropout5_mod/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
з
4gradients/dropout5_mod/cond/dropout/mul_grad/ReshapeReshape0gradients/dropout5_mod/cond/dropout/mul_grad/Sum2gradients/dropout5_mod/cond/dropout/mul_grad/Shape*5
_output_shapes#
!:                  ђ*
T0*
Tshape0
█
2gradients/dropout5_mod/cond/dropout/mul_grad/mul_1Muldropout5_mod/cond/dropout/divAgradients/dropout5_mod/cond/Merge_grad/tuple/control_dependency_1*5
_output_shapes#
!:                  ђ*
T0
з
2gradients/dropout5_mod/cond/dropout/mul_grad/Sum_1Sum2gradients/dropout5_mod/cond/dropout/mul_grad/mul_1Dgradients/dropout5_mod/cond/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
щ
6gradients/dropout5_mod/cond/dropout/mul_grad/Reshape_1Reshape2gradients/dropout5_mod/cond/dropout/mul_grad/Sum_14gradients/dropout5_mod/cond/dropout/mul_grad/Shape_1*5
_output_shapes#
!:                  ђ*
T0*
Tshape0
х
=gradients/dropout5_mod/cond/dropout/mul_grad/tuple/group_depsNoOp5^gradients/dropout5_mod/cond/dropout/mul_grad/Reshape7^gradients/dropout5_mod/cond/dropout/mul_grad/Reshape_1
л
Egradients/dropout5_mod/cond/dropout/mul_grad/tuple/control_dependencyIdentity4gradients/dropout5_mod/cond/dropout/mul_grad/Reshape>^gradients/dropout5_mod/cond/dropout/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/dropout5_mod/cond/dropout/mul_grad/Reshape*5
_output_shapes#
!:                  ђ
о
Ggradients/dropout5_mod/cond/dropout/mul_grad/tuple/control_dependency_1Identity6gradients/dropout5_mod/cond/dropout/mul_grad/Reshape_1>^gradients/dropout5_mod/cond/dropout/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/dropout5_mod/cond/dropout/mul_grad/Reshape_1*5
_output_shapes#
!:                  ђ
ц
gradients/SwitchSwitchfc5_mod/Reludropout5_mod/cond/pred_id*V
_output_shapesD
B:                  ђ:                  ђ*
T0
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
out_type0*
_output_shapes
:
Z
gradients/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Њ
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*

index_type0*5
_output_shapes#
!:                  ђ
Я
:gradients/dropout5_mod/cond/Identity/Switch_grad/cond_gradMerge?gradients/dropout5_mod/cond/Merge_grad/tuple/control_dependencygradients/zeros*
T0*
N*7
_output_shapes%
#:                  ђ: 
џ
2gradients/dropout5_mod/cond/dropout/div_grad/ShapeShape(dropout5_mod/cond/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0
w
4gradients/dropout5_mod/cond/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
ѓ
Bgradients/dropout5_mod/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/dropout5_mod/cond/dropout/div_grad/Shape4gradients/dropout5_mod/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
в
4gradients/dropout5_mod/cond/dropout/div_grad/RealDivRealDivEgradients/dropout5_mod/cond/dropout/mul_grad/tuple/control_dependency#dropout5_mod/cond/dropout/keep_prob*
T0*5
_output_shapes#
!:                  ђ
ы
0gradients/dropout5_mod/cond/dropout/div_grad/SumSum4gradients/dropout5_mod/cond/dropout/div_grad/RealDivBgradients/dropout5_mod/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
з
4gradients/dropout5_mod/cond/dropout/div_grad/ReshapeReshape0gradients/dropout5_mod/cond/dropout/div_grad/Sum2gradients/dropout5_mod/cond/dropout/div_grad/Shape*
T0*
Tshape0*5
_output_shapes#
!:                  ђ
А
0gradients/dropout5_mod/cond/dropout/div_grad/NegNeg(dropout5_mod/cond/dropout/Shape/Switch:1*
T0*5
_output_shapes#
!:                  ђ
п
6gradients/dropout5_mod/cond/dropout/div_grad/RealDiv_1RealDiv0gradients/dropout5_mod/cond/dropout/div_grad/Neg#dropout5_mod/cond/dropout/keep_prob*
T0*5
_output_shapes#
!:                  ђ
я
6gradients/dropout5_mod/cond/dropout/div_grad/RealDiv_2RealDiv6gradients/dropout5_mod/cond/dropout/div_grad/RealDiv_1#dropout5_mod/cond/dropout/keep_prob*
T0*5
_output_shapes#
!:                  ђ
Ш
0gradients/dropout5_mod/cond/dropout/div_grad/mulMulEgradients/dropout5_mod/cond/dropout/mul_grad/tuple/control_dependency6gradients/dropout5_mod/cond/dropout/div_grad/RealDiv_2*
T0*5
_output_shapes#
!:                  ђ
ы
2gradients/dropout5_mod/cond/dropout/div_grad/Sum_1Sum0gradients/dropout5_mod/cond/dropout/div_grad/mulDgradients/dropout5_mod/cond/dropout/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
┌
6gradients/dropout5_mod/cond/dropout/div_grad/Reshape_1Reshape2gradients/dropout5_mod/cond/dropout/div_grad/Sum_14gradients/dropout5_mod/cond/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
х
=gradients/dropout5_mod/cond/dropout/div_grad/tuple/group_depsNoOp5^gradients/dropout5_mod/cond/dropout/div_grad/Reshape7^gradients/dropout5_mod/cond/dropout/div_grad/Reshape_1
л
Egradients/dropout5_mod/cond/dropout/div_grad/tuple/control_dependencyIdentity4gradients/dropout5_mod/cond/dropout/div_grad/Reshape>^gradients/dropout5_mod/cond/dropout/div_grad/tuple/group_deps*5
_output_shapes#
!:                  ђ*
T0*G
_class=
;9loc:@gradients/dropout5_mod/cond/dropout/div_grad/Reshape
и
Ggradients/dropout5_mod/cond/dropout/div_grad/tuple/control_dependency_1Identity6gradients/dropout5_mod/cond/dropout/div_grad/Reshape_1>^gradients/dropout5_mod/cond/dropout/div_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/dropout5_mod/cond/dropout/div_grad/Reshape_1*
_output_shapes
: 
д
gradients/Switch_1Switchfc5_mod/Reludropout5_mod/cond/pred_id*
T0*V
_output_shapesD
B:                  ђ:                  ђ
c
gradients/Shape_2Shapegradients/Switch_1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ќ
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*

index_type0*5
_output_shapes#
!:                  ђ
ь
?gradients/dropout5_mod/cond/dropout/Shape/Switch_grad/cond_gradMergegradients/zeros_1Egradients/dropout5_mod/cond/dropout/div_grad/tuple/control_dependency*
N*7
_output_shapes%
#:                  ђ: *
T0
Г
gradients/AddN_1AddN:gradients/dropout5_mod/cond/Identity/Switch_grad/cond_grad?gradients/dropout5_mod/cond/dropout/Shape/Switch_grad/cond_grad*
N*5
_output_shapes#
!:                  ђ*
T0*M
_classC
A?loc:@gradients/dropout5_mod/cond/Identity/Switch_grad/cond_grad
љ
$gradients/fc5_mod/Relu_grad/ReluGradReluGradgradients/AddN_1fc5_mod/Relu*
T0*5
_output_shapes#
!:                  ђ
ю
*gradients/fc5_mod/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/fc5_mod/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:ђ
І
/gradients/fc5_mod/BiasAdd_grad/tuple/group_depsNoOp%^gradients/fc5_mod/Relu_grad/ReluGrad+^gradients/fc5_mod/BiasAdd_grad/BiasAddGrad
ћ
7gradients/fc5_mod/BiasAdd_grad/tuple/control_dependencyIdentity$gradients/fc5_mod/Relu_grad/ReluGrad0^gradients/fc5_mod/BiasAdd_grad/tuple/group_deps*5
_output_shapes#
!:                  ђ*
T0*7
_class-
+)loc:@gradients/fc5_mod/Relu_grad/ReluGrad
ѕ
9gradients/fc5_mod/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/fc5_mod/BiasAdd_grad/BiasAddGrad0^gradients/fc5_mod/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/fc5_mod/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
~
&gradients/fc5_mod/Tensordot_grad/ShapeShapefc5_mod/Tensordot/MatMul*
_output_shapes
:*
T0*
out_type0
Н
(gradients/fc5_mod/Tensordot_grad/ReshapeReshape7gradients/fc5_mod/BiasAdd_grad/tuple/control_dependency&gradients/fc5_mod/Tensordot_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
п
.gradients/fc5_mod/Tensordot/MatMul_grad/MatMulMatMul(gradients/fc5_mod/Tensordot_grad/Reshapefc5_mod/Tensordot/Reshape_1*
T0*(
_output_shapes
:         ђ,*
transpose_a( *
transpose_b(
п
0gradients/fc5_mod/Tensordot/MatMul_grad/MatMul_1MatMulfc5_mod/Tensordot/Reshape(gradients/fc5_mod/Tensordot_grad/Reshape*
T0*(
_output_shapes
:         ђ*
transpose_a(*
transpose_b( 
ц
8gradients/fc5_mod/Tensordot/MatMul_grad/tuple/group_depsNoOp/^gradients/fc5_mod/Tensordot/MatMul_grad/MatMul1^gradients/fc5_mod/Tensordot/MatMul_grad/MatMul_1
Г
@gradients/fc5_mod/Tensordot/MatMul_grad/tuple/control_dependencyIdentity.gradients/fc5_mod/Tensordot/MatMul_grad/MatMul9^gradients/fc5_mod/Tensordot/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ђ,*
T0*A
_class7
53loc:@gradients/fc5_mod/Tensordot/MatMul_grad/MatMul
Ф
Bgradients/fc5_mod/Tensordot/MatMul_grad/tuple/control_dependency_1Identity0gradients/fc5_mod/Tensordot/MatMul_grad/MatMul_19^gradients/fc5_mod/Tensordot/MatMul_grad/tuple/group_deps* 
_output_shapes
:
ђ,ђ*
T0*C
_class9
75loc:@gradients/fc5_mod/Tensordot/MatMul_grad/MatMul_1
Ѕ
.gradients/fc5_mod/Tensordot/Reshape_grad/ShapeShapefc5_mod/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
Ѓ
0gradients/fc5_mod/Tensordot/Reshape_grad/ReshapeReshape@gradients/fc5_mod/Tensordot/MatMul_grad/tuple/control_dependency.gradients/fc5_mod/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*=
_output_shapes+
):'                           
Ђ
0gradients/fc5_mod/Tensordot/Reshape_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
В
2gradients/fc5_mod/Tensordot/Reshape_1_grad/ReshapeReshapeBgradients/fc5_mod/Tensordot/MatMul_grad/tuple/control_dependency_10gradients/fc5_mod/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
ђ,ђ
Џ
<gradients/fc5_mod/Tensordot/transpose_grad/InvertPermutationInvertPermutationfc5_mod/Tensordot/concat_1*#
_output_shapes
:         *
T0
■
4gradients/fc5_mod/Tensordot/transpose_grad/transpose	Transpose0gradients/fc5_mod/Tensordot/Reshape_grad/Reshape<gradients/fc5_mod/Tensordot/transpose_grad/InvertPermutation*5
_output_shapes#
!:                  ђ,*
Tperm0*
T0
ю
>gradients/fc5_mod/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation"fc5_mod/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
№
6gradients/fc5_mod/Tensordot/transpose_1_grad/transpose	Transpose2gradients/fc5_mod/Tensordot/Reshape_1_grad/Reshape>gradients/fc5_mod/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0* 
_output_shapes
:
ђ,ђ
x
!gradients/flatten4_mod_grad/ShapeShapedropout3_mod/cond/Merge*
_output_shapes
:*
T0*
out_type0
п
#gradients/flatten4_mod_grad/ReshapeReshape4gradients/fc5_mod/Tensordot/transpose_grad/transpose!gradients/flatten4_mod_grad/Shape*
T0*
Tshape0*8
_output_shapes&
$:"                  X@
Ў
0gradients/dropout3_mod/cond/Merge_grad/cond_gradSwitch#gradients/flatten4_mod_grad/Reshapedropout3_mod/cond/pred_id*\
_output_shapesJ
H:"                  X@:"                  X@*
T0*6
_class,
*(loc:@gradients/flatten4_mod_grad/Reshape
r
7gradients/dropout3_mod/cond/Merge_grad/tuple/group_depsNoOp1^gradients/dropout3_mod/cond/Merge_grad/cond_grad
▓
?gradients/dropout3_mod/cond/Merge_grad/tuple/control_dependencyIdentity0gradients/dropout3_mod/cond/Merge_grad/cond_grad8^gradients/dropout3_mod/cond/Merge_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/flatten4_mod_grad/Reshape*8
_output_shapes&
$:"                  X@
Х
Agradients/dropout3_mod/cond/Merge_grad/tuple/control_dependency_1Identity2gradients/dropout3_mod/cond/Merge_grad/cond_grad:18^gradients/dropout3_mod/cond/Merge_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/flatten4_mod_grad/Reshape*8
_output_shapes&
$:"                  X@
Ј
2gradients/dropout3_mod/cond/dropout/mul_grad/ShapeShapedropout3_mod/cond/dropout/div*
T0*
out_type0*
_output_shapes
:
Њ
4gradients/dropout3_mod/cond/dropout/mul_grad/Shape_1Shapedropout3_mod/cond/dropout/Floor*
T0*
out_type0*
_output_shapes
:
ѓ
Bgradients/dropout3_mod/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/dropout3_mod/cond/dropout/mul_grad/Shape4gradients/dropout3_mod/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
я
0gradients/dropout3_mod/cond/dropout/mul_grad/mulMulAgradients/dropout3_mod/cond/Merge_grad/tuple/control_dependency_1dropout3_mod/cond/dropout/Floor*
T0*8
_output_shapes&
$:"                  X@
ь
0gradients/dropout3_mod/cond/dropout/mul_grad/SumSum0gradients/dropout3_mod/cond/dropout/mul_grad/mulBgradients/dropout3_mod/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ш
4gradients/dropout3_mod/cond/dropout/mul_grad/ReshapeReshape0gradients/dropout3_mod/cond/dropout/mul_grad/Sum2gradients/dropout3_mod/cond/dropout/mul_grad/Shape*8
_output_shapes&
$:"                  X@*
T0*
Tshape0
я
2gradients/dropout3_mod/cond/dropout/mul_grad/mul_1Muldropout3_mod/cond/dropout/divAgradients/dropout3_mod/cond/Merge_grad/tuple/control_dependency_1*
T0*8
_output_shapes&
$:"                  X@
з
2gradients/dropout3_mod/cond/dropout/mul_grad/Sum_1Sum2gradients/dropout3_mod/cond/dropout/mul_grad/mul_1Dgradients/dropout3_mod/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ч
6gradients/dropout3_mod/cond/dropout/mul_grad/Reshape_1Reshape2gradients/dropout3_mod/cond/dropout/mul_grad/Sum_14gradients/dropout3_mod/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*8
_output_shapes&
$:"                  X@
х
=gradients/dropout3_mod/cond/dropout/mul_grad/tuple/group_depsNoOp5^gradients/dropout3_mod/cond/dropout/mul_grad/Reshape7^gradients/dropout3_mod/cond/dropout/mul_grad/Reshape_1
М
Egradients/dropout3_mod/cond/dropout/mul_grad/tuple/control_dependencyIdentity4gradients/dropout3_mod/cond/dropout/mul_grad/Reshape>^gradients/dropout3_mod/cond/dropout/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/dropout3_mod/cond/dropout/mul_grad/Reshape*8
_output_shapes&
$:"                  X@
┘
Ggradients/dropout3_mod/cond/dropout/mul_grad/tuple/control_dependency_1Identity6gradients/dropout3_mod/cond/dropout/mul_grad/Reshape_1>^gradients/dropout3_mod/cond/dropout/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/dropout3_mod/cond/dropout/mul_grad/Reshape_1*8
_output_shapes&
$:"                  X@
▒
gradients/Switch_2Switchpool3_mod/MaxPooldropout3_mod/cond/pred_id*\
_output_shapesJ
H:"                  X@:"                  X@*
T0
e
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
џ
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*8
_output_shapes&
$:"                  X@*
T0*

index_type0
т
:gradients/dropout3_mod/cond/Identity/Switch_grad/cond_gradMerge?gradients/dropout3_mod/cond/Merge_grad/tuple/control_dependencygradients/zeros_2*
T0*
N*:
_output_shapes(
&:"                  X@: 
џ
2gradients/dropout3_mod/cond/dropout/div_grad/ShapeShape(dropout3_mod/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
w
4gradients/dropout3_mod/cond/dropout/div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
ѓ
Bgradients/dropout3_mod/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/dropout3_mod/cond/dropout/div_grad/Shape4gradients/dropout3_mod/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ь
4gradients/dropout3_mod/cond/dropout/div_grad/RealDivRealDivEgradients/dropout3_mod/cond/dropout/mul_grad/tuple/control_dependency#dropout3_mod/cond/dropout/keep_prob*8
_output_shapes&
$:"                  X@*
T0
ы
0gradients/dropout3_mod/cond/dropout/div_grad/SumSum4gradients/dropout3_mod/cond/dropout/div_grad/RealDivBgradients/dropout3_mod/cond/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ш
4gradients/dropout3_mod/cond/dropout/div_grad/ReshapeReshape0gradients/dropout3_mod/cond/dropout/div_grad/Sum2gradients/dropout3_mod/cond/dropout/div_grad/Shape*
T0*
Tshape0*8
_output_shapes&
$:"                  X@
ц
0gradients/dropout3_mod/cond/dropout/div_grad/NegNeg(dropout3_mod/cond/dropout/Shape/Switch:1*
T0*8
_output_shapes&
$:"                  X@
█
6gradients/dropout3_mod/cond/dropout/div_grad/RealDiv_1RealDiv0gradients/dropout3_mod/cond/dropout/div_grad/Neg#dropout3_mod/cond/dropout/keep_prob*8
_output_shapes&
$:"                  X@*
T0
р
6gradients/dropout3_mod/cond/dropout/div_grad/RealDiv_2RealDiv6gradients/dropout3_mod/cond/dropout/div_grad/RealDiv_1#dropout3_mod/cond/dropout/keep_prob*
T0*8
_output_shapes&
$:"                  X@
щ
0gradients/dropout3_mod/cond/dropout/div_grad/mulMulEgradients/dropout3_mod/cond/dropout/mul_grad/tuple/control_dependency6gradients/dropout3_mod/cond/dropout/div_grad/RealDiv_2*
T0*8
_output_shapes&
$:"                  X@
ы
2gradients/dropout3_mod/cond/dropout/div_grad/Sum_1Sum0gradients/dropout3_mod/cond/dropout/div_grad/mulDgradients/dropout3_mod/cond/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
┌
6gradients/dropout3_mod/cond/dropout/div_grad/Reshape_1Reshape2gradients/dropout3_mod/cond/dropout/div_grad/Sum_14gradients/dropout3_mod/cond/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
х
=gradients/dropout3_mod/cond/dropout/div_grad/tuple/group_depsNoOp5^gradients/dropout3_mod/cond/dropout/div_grad/Reshape7^gradients/dropout3_mod/cond/dropout/div_grad/Reshape_1
М
Egradients/dropout3_mod/cond/dropout/div_grad/tuple/control_dependencyIdentity4gradients/dropout3_mod/cond/dropout/div_grad/Reshape>^gradients/dropout3_mod/cond/dropout/div_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/dropout3_mod/cond/dropout/div_grad/Reshape*8
_output_shapes&
$:"                  X@
и
Ggradients/dropout3_mod/cond/dropout/div_grad/tuple/control_dependency_1Identity6gradients/dropout3_mod/cond/dropout/div_grad/Reshape_1>^gradients/dropout3_mod/cond/dropout/div_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/dropout3_mod/cond/dropout/div_grad/Reshape_1*
_output_shapes
: 
▒
gradients/Switch_3Switchpool3_mod/MaxPooldropout3_mod/cond/pred_id*
T0*\
_output_shapesJ
H:"                  X@:"                  X@
c
gradients/Shape_4Shapegradients/Switch_3*
_output_shapes
:*
T0*
out_type0
\
gradients/zeros_3/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
џ
gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*

index_type0*8
_output_shapes&
$:"                  X@
­
?gradients/dropout3_mod/cond/dropout/Shape/Switch_grad/cond_gradMergegradients/zeros_3Egradients/dropout3_mod/cond/dropout/div_grad/tuple/control_dependency*
N*:
_output_shapes(
&:"                  X@: *
T0
░
gradients/AddN_2AddN:gradients/dropout3_mod/cond/Identity/Switch_grad/cond_grad?gradients/dropout3_mod/cond/dropout/Shape/Switch_grad/cond_grad*
N*8
_output_shapes&
$:"                  X@*
T0*M
_classC
A?loc:@gradients/dropout3_mod/cond/Identity/Switch_grad/cond_grad
ў
,gradients/pool3_mod/MaxPool_grad/MaxPoolGradMaxPoolGradharmonic_layer_1/conv3_mod/Relupool3_mod/MaxPoolgradients/AddN_2*
ksize
*
paddingVALID*9
_output_shapes'
%:#                  ░@*
T0*
data_formatNHWC*
strides

о
7gradients/harmonic_layer_1/conv3_mod/Relu_grad/ReluGradReluGrad,gradients/pool3_mod/MaxPool_grad/MaxPoolGradharmonic_layer_1/conv3_mod/Relu*
T0*9
_output_shapes'
%:#                  ░@
┴
=gradients/harmonic_layer_1/conv3_mod/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients/harmonic_layer_1/conv3_mod/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
─
Bgradients/harmonic_layer_1/conv3_mod/BiasAdd_grad/tuple/group_depsNoOp8^gradients/harmonic_layer_1/conv3_mod/Relu_grad/ReluGrad>^gradients/harmonic_layer_1/conv3_mod/BiasAdd_grad/BiasAddGrad
С
Jgradients/harmonic_layer_1/conv3_mod/BiasAdd_grad/tuple/control_dependencyIdentity7gradients/harmonic_layer_1/conv3_mod/Relu_grad/ReluGradC^gradients/harmonic_layer_1/conv3_mod/BiasAdd_grad/tuple/group_deps*9
_output_shapes'
%:#                  ░@*
T0*J
_class@
><loc:@gradients/harmonic_layer_1/conv3_mod/Relu_grad/ReluGrad
М
Lgradients/harmonic_layer_1/conv3_mod/BiasAdd_grad/tuple/control_dependency_1Identity=gradients/harmonic_layer_1/conv3_mod/BiasAdd_grad/BiasAddGradC^gradients/harmonic_layer_1/conv3_mod/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/harmonic_layer_1/conv3_mod/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
╗
7gradients/harmonic_layer_1/conv3_mod/Conv2D_grad/ShapeNShapeNharmonic_layer_1/transpose_1conv3_mod/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ј
6gradients/harmonic_layer_1/conv3_mod/Conv2D_grad/ConstConst*%
valueB"          @   *
dtype0*
_output_shapes
:
╝
Dgradients/harmonic_layer_1/conv3_mod/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7gradients/harmonic_layer_1/conv3_mod/Conv2D_grad/ShapeNconv3_mod/weights/readJgradients/harmonic_layer_1/conv3_mod/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
Ъ
Egradients/harmonic_layer_1/conv3_mod/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterharmonic_layer_1/transpose_16gradients/harmonic_layer_1/conv3_mod/Conv2D_grad/ConstJgradients/harmonic_layer_1/conv3_mod/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
: @
п
Agradients/harmonic_layer_1/conv3_mod/Conv2D_grad/tuple/group_depsNoOpE^gradients/harmonic_layer_1/conv3_mod/Conv2D_grad/Conv2DBackpropInputF^gradients/harmonic_layer_1/conv3_mod/Conv2D_grad/Conv2DBackpropFilter
Ч
Igradients/harmonic_layer_1/conv3_mod/Conv2D_grad/tuple/control_dependencyIdentityDgradients/harmonic_layer_1/conv3_mod/Conv2D_grad/Conv2DBackpropInputB^gradients/harmonic_layer_1/conv3_mod/Conv2D_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/harmonic_layer_1/conv3_mod/Conv2D_grad/Conv2DBackpropInput*9
_output_shapes'
%:#                  л	 
ь
Kgradients/harmonic_layer_1/conv3_mod/Conv2D_grad/tuple/control_dependency_1IdentityEgradients/harmonic_layer_1/conv3_mod/Conv2D_grad/Conv2DBackpropFilterB^gradients/harmonic_layer_1/conv3_mod/Conv2D_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/harmonic_layer_1/conv3_mod/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
џ
=gradients/harmonic_layer_1/transpose_1_grad/InvertPermutationInvertPermutation!harmonic_layer_1/transpose_1/perm*
T0*
_output_shapes
:
Ю
5gradients/harmonic_layer_1/transpose_1_grad/transpose	TransposeIgradients/harmonic_layer_1/conv3_mod/Conv2D_grad/tuple/control_dependency=gradients/harmonic_layer_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*9
_output_shapes'
%:#л	                   
і
0gradients/harmonic_layer_1/Reordering_grad/ShapeShapeharmonic_layer_1/transpose*
T0*
out_type0*
_output_shapes
:
е
2gradients/harmonic_layer_1/Reordering_grad/SqueezeSqueeze#harmonic_layer_1/Reordering/indices*
_output_shapes	
:л	*
squeeze_dims

         *
T0
ќ
;gradients/harmonic_layer_1/transpose_grad/InvertPermutationInvertPermutationharmonic_layer_1/transpose/perm*
T0*
_output_shapes
:
Љ
Ggradients/harmonic_layer_1/transpose_grad/transpose/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Њ
Igradients/harmonic_layer_1/transpose_grad/transpose/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Њ
Igradients/harmonic_layer_1/transpose_grad/transpose/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
З
Agradients/harmonic_layer_1/transpose_grad/transpose/strided_sliceStridedSlice0gradients/harmonic_layer_1/Reordering_grad/ShapeGgradients/harmonic_layer_1/transpose_grad/transpose/strided_slice/stackIgradients/harmonic_layer_1/transpose_grad/transpose/strided_slice/stack_1Igradients/harmonic_layer_1/transpose_grad/transpose/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
ж
5gradients/harmonic_layer_1/transpose_grad/transpose/xUnsortedSegmentSum5gradients/harmonic_layer_1/transpose_1_grad/transpose2gradients/harmonic_layer_1/Reordering_grad/SqueezeAgradients/harmonic_layer_1/transpose_grad/transpose/strided_slice*
Tnumsegments0*
Tindices0*
T0*A
_output_shapes/
-:+                            
Ё
3gradients/harmonic_layer_1/transpose_grad/transpose	Transpose5gradients/harmonic_layer_1/transpose_grad/transpose/x;gradients/harmonic_layer_1/transpose_grad/InvertPermutation*
T0*9
_output_shapes'
%:#                  ▒ *
Tperm0
j
(gradients/harmonic_layer_1/Pad_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
m
+gradients/harmonic_layer_1/Pad_grad/stack/1Const*
dtype0*
_output_shapes
: *
value	B :
┬
)gradients/harmonic_layer_1/Pad_grad/stackPack(gradients/harmonic_layer_1/Pad_grad/Rank+gradients/harmonic_layer_1/Pad_grad/stack/1*
T0*

axis *
N*
_output_shapes
:
ђ
/gradients/harmonic_layer_1/Pad_grad/Slice/beginConst*
valueB"        *
dtype0*
_output_shapes
:
▄
)gradients/harmonic_layer_1/Pad_grad/SliceSliceharmonic_layer_1/Const/gradients/harmonic_layer_1/Pad_grad/Slice/begin)gradients/harmonic_layer_1/Pad_grad/stack*
_output_shapes

:*
T0*
Index0
ё
1gradients/harmonic_layer_1/Pad_grad/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
К
+gradients/harmonic_layer_1/Pad_grad/ReshapeReshape)gradients/harmonic_layer_1/Pad_grad/Slice1gradients/harmonic_layer_1/Pad_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
ђ
)gradients/harmonic_layer_1/Pad_grad/ShapeShapedropout2_mod/cond/Merge*
T0*
out_type0*
_output_shapes
:
њ
+gradients/harmonic_layer_1/Pad_grad/Slice_1Slice3gradients/harmonic_layer_1/transpose_grad/transpose+gradients/harmonic_layer_1/Pad_grad/Reshape)gradients/harmonic_layer_1/Pad_grad/Shape*
T0*
Index0*9
_output_shapes'
%:#                  ░ 
Ф
0gradients/dropout2_mod/cond/Merge_grad/cond_gradSwitch+gradients/harmonic_layer_1/Pad_grad/Slice_1dropout2_mod/cond/pred_id*
T0*>
_class4
20loc:@gradients/harmonic_layer_1/Pad_grad/Slice_1*^
_output_shapesL
J:#                  ░ :#                  ░ 
r
7gradients/dropout2_mod/cond/Merge_grad/tuple/group_depsNoOp1^gradients/dropout2_mod/cond/Merge_grad/cond_grad
╗
?gradients/dropout2_mod/cond/Merge_grad/tuple/control_dependencyIdentity0gradients/dropout2_mod/cond/Merge_grad/cond_grad8^gradients/dropout2_mod/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/harmonic_layer_1/Pad_grad/Slice_1*9
_output_shapes'
%:#                  ░ 
┐
Agradients/dropout2_mod/cond/Merge_grad/tuple/control_dependency_1Identity2gradients/dropout2_mod/cond/Merge_grad/cond_grad:18^gradients/dropout2_mod/cond/Merge_grad/tuple/group_deps*9
_output_shapes'
%:#                  ░ *
T0*>
_class4
20loc:@gradients/harmonic_layer_1/Pad_grad/Slice_1
Ј
2gradients/dropout2_mod/cond/dropout/mul_grad/ShapeShapedropout2_mod/cond/dropout/div*
T0*
out_type0*
_output_shapes
:
Њ
4gradients/dropout2_mod/cond/dropout/mul_grad/Shape_1Shapedropout2_mod/cond/dropout/Floor*
T0*
out_type0*
_output_shapes
:
ѓ
Bgradients/dropout2_mod/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/dropout2_mod/cond/dropout/mul_grad/Shape4gradients/dropout2_mod/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
▀
0gradients/dropout2_mod/cond/dropout/mul_grad/mulMulAgradients/dropout2_mod/cond/Merge_grad/tuple/control_dependency_1dropout2_mod/cond/dropout/Floor*
T0*9
_output_shapes'
%:#                  ░ 
ь
0gradients/dropout2_mod/cond/dropout/mul_grad/SumSum0gradients/dropout2_mod/cond/dropout/mul_grad/mulBgradients/dropout2_mod/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
э
4gradients/dropout2_mod/cond/dropout/mul_grad/ReshapeReshape0gradients/dropout2_mod/cond/dropout/mul_grad/Sum2gradients/dropout2_mod/cond/dropout/mul_grad/Shape*
T0*
Tshape0*9
_output_shapes'
%:#                  ░ 
▀
2gradients/dropout2_mod/cond/dropout/mul_grad/mul_1Muldropout2_mod/cond/dropout/divAgradients/dropout2_mod/cond/Merge_grad/tuple/control_dependency_1*9
_output_shapes'
%:#                  ░ *
T0
з
2gradients/dropout2_mod/cond/dropout/mul_grad/Sum_1Sum2gradients/dropout2_mod/cond/dropout/mul_grad/mul_1Dgradients/dropout2_mod/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
§
6gradients/dropout2_mod/cond/dropout/mul_grad/Reshape_1Reshape2gradients/dropout2_mod/cond/dropout/mul_grad/Sum_14gradients/dropout2_mod/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*9
_output_shapes'
%:#                  ░ 
х
=gradients/dropout2_mod/cond/dropout/mul_grad/tuple/group_depsNoOp5^gradients/dropout2_mod/cond/dropout/mul_grad/Reshape7^gradients/dropout2_mod/cond/dropout/mul_grad/Reshape_1
н
Egradients/dropout2_mod/cond/dropout/mul_grad/tuple/control_dependencyIdentity4gradients/dropout2_mod/cond/dropout/mul_grad/Reshape>^gradients/dropout2_mod/cond/dropout/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/dropout2_mod/cond/dropout/mul_grad/Reshape*9
_output_shapes'
%:#                  ░ 
┌
Ggradients/dropout2_mod/cond/dropout/mul_grad/tuple/control_dependency_1Identity6gradients/dropout2_mod/cond/dropout/mul_grad/Reshape_1>^gradients/dropout2_mod/cond/dropout/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/dropout2_mod/cond/dropout/mul_grad/Reshape_1*9
_output_shapes'
%:#                  ░ 
│
gradients/Switch_4Switchpool2_mod/MaxPooldropout2_mod/cond/pred_id*
T0*^
_output_shapesL
J:#                  ░ :#                  ░ 
e
gradients/Shape_5Shapegradients/Switch_4:1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Џ
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*

index_type0*9
_output_shapes'
%:#                  ░ 
Т
:gradients/dropout2_mod/cond/Identity/Switch_grad/cond_gradMerge?gradients/dropout2_mod/cond/Merge_grad/tuple/control_dependencygradients/zeros_4*
N*;
_output_shapes)
':#                  ░ : *
T0
џ
2gradients/dropout2_mod/cond/dropout/div_grad/ShapeShape(dropout2_mod/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
w
4gradients/dropout2_mod/cond/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
ѓ
Bgradients/dropout2_mod/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/dropout2_mod/cond/dropout/div_grad/Shape4gradients/dropout2_mod/cond/dropout/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
№
4gradients/dropout2_mod/cond/dropout/div_grad/RealDivRealDivEgradients/dropout2_mod/cond/dropout/mul_grad/tuple/control_dependency#dropout2_mod/cond/dropout/keep_prob*
T0*9
_output_shapes'
%:#                  ░ 
ы
0gradients/dropout2_mod/cond/dropout/div_grad/SumSum4gradients/dropout2_mod/cond/dropout/div_grad/RealDivBgradients/dropout2_mod/cond/dropout/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
э
4gradients/dropout2_mod/cond/dropout/div_grad/ReshapeReshape0gradients/dropout2_mod/cond/dropout/div_grad/Sum2gradients/dropout2_mod/cond/dropout/div_grad/Shape*
T0*
Tshape0*9
_output_shapes'
%:#                  ░ 
Ц
0gradients/dropout2_mod/cond/dropout/div_grad/NegNeg(dropout2_mod/cond/dropout/Shape/Switch:1*
T0*9
_output_shapes'
%:#                  ░ 
▄
6gradients/dropout2_mod/cond/dropout/div_grad/RealDiv_1RealDiv0gradients/dropout2_mod/cond/dropout/div_grad/Neg#dropout2_mod/cond/dropout/keep_prob*
T0*9
_output_shapes'
%:#                  ░ 
Р
6gradients/dropout2_mod/cond/dropout/div_grad/RealDiv_2RealDiv6gradients/dropout2_mod/cond/dropout/div_grad/RealDiv_1#dropout2_mod/cond/dropout/keep_prob*
T0*9
_output_shapes'
%:#                  ░ 
Щ
0gradients/dropout2_mod/cond/dropout/div_grad/mulMulEgradients/dropout2_mod/cond/dropout/mul_grad/tuple/control_dependency6gradients/dropout2_mod/cond/dropout/div_grad/RealDiv_2*9
_output_shapes'
%:#                  ░ *
T0
ы
2gradients/dropout2_mod/cond/dropout/div_grad/Sum_1Sum0gradients/dropout2_mod/cond/dropout/div_grad/mulDgradients/dropout2_mod/cond/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
┌
6gradients/dropout2_mod/cond/dropout/div_grad/Reshape_1Reshape2gradients/dropout2_mod/cond/dropout/div_grad/Sum_14gradients/dropout2_mod/cond/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
х
=gradients/dropout2_mod/cond/dropout/div_grad/tuple/group_depsNoOp5^gradients/dropout2_mod/cond/dropout/div_grad/Reshape7^gradients/dropout2_mod/cond/dropout/div_grad/Reshape_1
н
Egradients/dropout2_mod/cond/dropout/div_grad/tuple/control_dependencyIdentity4gradients/dropout2_mod/cond/dropout/div_grad/Reshape>^gradients/dropout2_mod/cond/dropout/div_grad/tuple/group_deps*9
_output_shapes'
%:#                  ░ *
T0*G
_class=
;9loc:@gradients/dropout2_mod/cond/dropout/div_grad/Reshape
и
Ggradients/dropout2_mod/cond/dropout/div_grad/tuple/control_dependency_1Identity6gradients/dropout2_mod/cond/dropout/div_grad/Reshape_1>^gradients/dropout2_mod/cond/dropout/div_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/dropout2_mod/cond/dropout/div_grad/Reshape_1*
_output_shapes
: 
│
gradients/Switch_5Switchpool2_mod/MaxPooldropout2_mod/cond/pred_id*
T0*^
_output_shapesL
J:#                  ░ :#                  ░ 
c
gradients/Shape_6Shapegradients/Switch_5*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_5/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Џ
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
T0*

index_type0*9
_output_shapes'
%:#                  ░ 
ы
?gradients/dropout2_mod/cond/dropout/Shape/Switch_grad/cond_gradMergegradients/zeros_5Egradients/dropout2_mod/cond/dropout/div_grad/tuple/control_dependency*
T0*
N*;
_output_shapes)
':#                  ░ : 
▒
gradients/AddN_3AddN:gradients/dropout2_mod/cond/Identity/Switch_grad/cond_grad?gradients/dropout2_mod/cond/dropout/Shape/Switch_grad/cond_grad*
T0*M
_classC
A?loc:@gradients/dropout2_mod/cond/Identity/Switch_grad/cond_grad*
N*9
_output_shapes'
%:#                  ░ 
ќ
,gradients/pool2_mod/MaxPool_grad/MaxPoolGradMaxPoolGradharmonic_layer/conv2_mod/Relupool2_mod/MaxPoolgradients/AddN_3*
ksize
*
paddingVALID*9
_output_shapes'
%:#                  Я *
T0*
data_formatNHWC*
strides

м
5gradients/harmonic_layer/conv2_mod/Relu_grad/ReluGradReluGrad,gradients/pool2_mod/MaxPool_grad/MaxPoolGradharmonic_layer/conv2_mod/Relu*
T0*9
_output_shapes'
%:#                  Я 
{
gradients/zeros_like	ZerosLike3harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm:1*
T0*
_output_shapes
: 
}
gradients/zeros_like_1	ZerosLike3harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm:2*
T0*
_output_shapes
: 
}
gradients/zeros_like_2	ZerosLike3harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm:3*
T0*
_output_shapes
: 
}
gradients/zeros_like_3	ZerosLike3harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm:4*
_output_shapes
: *
T0
ж
Sgradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGrad5gradients/harmonic_layer/conv2_mod/Relu_grad/ReluGradharmonic_layer/conv2_mod/Conv2D(harmonic_layer/conv2_mod/BatchNorm/Const3harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm:33harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm:4*
T0*
data_formatNHWC*M
_output_shapes;
9:#                  Я : : : : *
is_training(*
epsilon%oЃ:
»
Qgradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/group_depsNoOpT^gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad
║
Ygradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/control_dependencyIdentitySgradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGradR^gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/group_deps*9
_output_shapes'
%:#                  Я *
T0*f
_class\
ZXloc:@gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad
Ъ
[gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency_1IdentityUgradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad:1R^gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
Ъ
[gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency_2IdentityUgradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad:2R^gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
Ю
[gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency_3IdentityUgradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad:3R^gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
Ю
[gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency_4IdentityUgradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad:4R^gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
и
5gradients/harmonic_layer/conv2_mod/Conv2D_grad/ShapeNShapeNharmonic_layer/transpose_1conv2_mod/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
Ї
4gradients/harmonic_layer/conv2_mod/Conv2D_grad/ConstConst*%
valueB"              *
dtype0*
_output_shapes
:
К
Bgradients/harmonic_layer/conv2_mod/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5gradients/harmonic_layer/conv2_mod/Conv2D_grad/ShapeNconv2_mod/weights/readYgradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency*
paddingVALID*J
_output_shapes8
6:4                                    *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
е
Cgradients/harmonic_layer/conv2_mod/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterharmonic_layer/transpose_14gradients/harmonic_layer/conv2_mod/Conv2D_grad/ConstYgradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:  *
	dilations

м
?gradients/harmonic_layer/conv2_mod/Conv2D_grad/tuple/group_depsNoOpC^gradients/harmonic_layer/conv2_mod/Conv2D_grad/Conv2DBackpropInputD^gradients/harmonic_layer/conv2_mod/Conv2D_grad/Conv2DBackpropFilter
З
Ggradients/harmonic_layer/conv2_mod/Conv2D_grad/tuple/control_dependencyIdentityBgradients/harmonic_layer/conv2_mod/Conv2D_grad/Conv2DBackpropInput@^gradients/harmonic_layer/conv2_mod/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/harmonic_layer/conv2_mod/Conv2D_grad/Conv2DBackpropInput*9
_output_shapes'
%:#                  а 
т
Igradients/harmonic_layer/conv2_mod/Conv2D_grad/tuple/control_dependency_1IdentityCgradients/harmonic_layer/conv2_mod/Conv2D_grad/Conv2DBackpropFilter@^gradients/harmonic_layer/conv2_mod/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/harmonic_layer/conv2_mod/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:  
ќ
;gradients/harmonic_layer/transpose_1_grad/InvertPermutationInvertPermutationharmonic_layer/transpose_1/perm*
_output_shapes
:*
T0
Ќ
3gradients/harmonic_layer/transpose_1_grad/transpose	TransposeGgradients/harmonic_layer/conv2_mod/Conv2D_grad/tuple/control_dependency;gradients/harmonic_layer/transpose_1_grad/InvertPermutation*9
_output_shapes'
%:#а                   *
Tperm0*
T0
є
.gradients/harmonic_layer/Reordering_grad/ShapeShapeharmonic_layer/transpose*
T0*
out_type0*
_output_shapes
:
ц
0gradients/harmonic_layer/Reordering_grad/SqueezeSqueeze!harmonic_layer/Reordering/indices*
T0*
_output_shapes	
:а*
squeeze_dims

         
њ
9gradients/harmonic_layer/transpose_grad/InvertPermutationInvertPermutationharmonic_layer/transpose/perm*
T0*
_output_shapes
:
Ј
Egradients/harmonic_layer/transpose_grad/transpose/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Љ
Ggradients/harmonic_layer/transpose_grad/transpose/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Љ
Ggradients/harmonic_layer/transpose_grad/transpose/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ж
?gradients/harmonic_layer/transpose_grad/transpose/strided_sliceStridedSlice.gradients/harmonic_layer/Reordering_grad/ShapeEgradients/harmonic_layer/transpose_grad/transpose/strided_slice/stackGgradients/harmonic_layer/transpose_grad/transpose/strided_slice/stack_1Ggradients/harmonic_layer/transpose_grad/transpose/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
р
3gradients/harmonic_layer/transpose_grad/transpose/xUnsortedSegmentSum3gradients/harmonic_layer/transpose_1_grad/transpose0gradients/harmonic_layer/Reordering_grad/Squeeze?gradients/harmonic_layer/transpose_grad/transpose/strided_slice*
Tnumsegments0*
Tindices0*
T0*A
_output_shapes/
-:+                            
 
1gradients/harmonic_layer/transpose_grad/transpose	Transpose3gradients/harmonic_layer/transpose_grad/transpose/x9gradients/harmonic_layer/transpose_grad/InvertPermutation*9
_output_shapes'
%:#                  р *
Tperm0*
T0
h
&gradients/harmonic_layer/Pad_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
k
)gradients/harmonic_layer/Pad_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 
╝
'gradients/harmonic_layer/Pad_grad/stackPack&gradients/harmonic_layer/Pad_grad/Rank)gradients/harmonic_layer/Pad_grad/stack/1*
N*
_output_shapes
:*
T0*

axis 
~
-gradients/harmonic_layer/Pad_grad/Slice/beginConst*
valueB"        *
dtype0*
_output_shapes
:
н
'gradients/harmonic_layer/Pad_grad/SliceSliceharmonic_layer/Const-gradients/harmonic_layer/Pad_grad/Slice/begin'gradients/harmonic_layer/Pad_grad/stack*
_output_shapes

:*
T0*
Index0
ѓ
/gradients/harmonic_layer/Pad_grad/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
┴
)gradients/harmonic_layer/Pad_grad/ReshapeReshape'gradients/harmonic_layer/Pad_grad/Slice/gradients/harmonic_layer/Pad_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
u
'gradients/harmonic_layer/Pad_grad/ShapeShapeconv1_mod/Relu*
T0*
out_type0*
_output_shapes
:
і
)gradients/harmonic_layer/Pad_grad/Slice_1Slice1gradients/harmonic_layer/transpose_grad/transpose)gradients/harmonic_layer/Pad_grad/Reshape'gradients/harmonic_layer/Pad_grad/Shape*9
_output_shapes'
%:#                  Я *
T0*
Index0
▒
&gradients/conv1_mod/Relu_grad/ReluGradReluGrad)gradients/harmonic_layer/Pad_grad/Slice_1conv1_mod/Relu*
T0*9
_output_shapes'
%:#                  Я 
Ъ
,gradients/conv1_mod/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients/conv1_mod/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
Љ
1gradients/conv1_mod/BiasAdd_grad/tuple/group_depsNoOp'^gradients/conv1_mod/Relu_grad/ReluGrad-^gradients/conv1_mod/BiasAdd_grad/BiasAddGrad
а
9gradients/conv1_mod/BiasAdd_grad/tuple/control_dependencyIdentity&gradients/conv1_mod/Relu_grad/ReluGrad2^gradients/conv1_mod/BiasAdd_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/conv1_mod/Relu_grad/ReluGrad*9
_output_shapes'
%:#                  Я 
Ј
;gradients/conv1_mod/BiasAdd_grad/tuple/control_dependency_1Identity,gradients/conv1_mod/BiasAdd_grad/BiasAddGrad2^gradients/conv1_mod/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv1_mod/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
Њ
&gradients/conv1_mod/Conv2D_grad/ShapeNShapeNinputconv1_mod/weights/read*
T0*
out_type0*
N* 
_output_shapes
::
~
%gradients/conv1_mod/Conv2D_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             
ѕ
3gradients/conv1_mod/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput&gradients/conv1_mod/Conv2D_grad/ShapeNconv1_mod/weights/read9gradients/conv1_mod/BiasAdd_grad/tuple/control_dependency*J
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
н
4gradients/conv1_mod/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput%gradients/conv1_mod/Conv2D_grad/Const9gradients/conv1_mod/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: 
Ц
0gradients/conv1_mod/Conv2D_grad/tuple/group_depsNoOp4^gradients/conv1_mod/Conv2D_grad/Conv2DBackpropInput5^gradients/conv1_mod/Conv2D_grad/Conv2DBackpropFilter
И
8gradients/conv1_mod/Conv2D_grad/tuple/control_dependencyIdentity3gradients/conv1_mod/Conv2D_grad/Conv2DBackpropInput1^gradients/conv1_mod/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv1_mod/Conv2D_grad/Conv2DBackpropInput*9
_output_shapes'
%:#                  Я
Е
:gradients/conv1_mod/Conv2D_grad/tuple/control_dependency_1Identity4gradients/conv1_mod/Conv2D_grad/Conv2DBackpropFilter1^gradients/conv1_mod/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/conv1_mod/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
Ѓ
beta1_power/initial_valueConst*
valueB
 *fff?*#
_class
loc:@conv1_mod/biases*
dtype0*
_output_shapes
: 
ћ
beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *#
_class
loc:@conv1_mod/biases
│
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*#
_class
loc:@conv1_mod/biases
o
beta1_power/readIdentitybeta1_power*
T0*#
_class
loc:@conv1_mod/biases*
_output_shapes
: 
Ѓ
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wЙ?*#
_class
loc:@conv1_mod/biases
ћ
beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *#
_class
loc:@conv1_mod/biases*
	container *
shape: 
│
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*#
_class
loc:@conv1_mod/biases*
validate_shape(*
_output_shapes
: 
o
beta2_power/readIdentitybeta2_power*
T0*#
_class
loc:@conv1_mod/biases*
_output_shapes
: 
и
8conv1_mod/weights/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"             *$
_class
loc:@conv1_mod/weights
Ў
.conv1_mod/weights/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *$
_class
loc:@conv1_mod/weights
Ѓ
(conv1_mod/weights/Adam/Initializer/zerosFill8conv1_mod/weights/Adam/Initializer/zeros/shape_as_tensor.conv1_mod/weights/Adam/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@conv1_mod/weights*&
_output_shapes
: 
└
conv1_mod/weights/Adam
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *$
_class
loc:@conv1_mod/weights*
	container *
shape: 
ж
conv1_mod/weights/Adam/AssignAssignconv1_mod/weights/Adam(conv1_mod/weights/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@conv1_mod/weights*
validate_shape(*&
_output_shapes
: 
ќ
conv1_mod/weights/Adam/readIdentityconv1_mod/weights/Adam*
T0*$
_class
loc:@conv1_mod/weights*&
_output_shapes
: 
╣
:conv1_mod/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"             *$
_class
loc:@conv1_mod/weights*
dtype0*
_output_shapes
:
Џ
0conv1_mod/weights/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@conv1_mod/weights*
dtype0*
_output_shapes
: 
Ѕ
*conv1_mod/weights/Adam_1/Initializer/zerosFill:conv1_mod/weights/Adam_1/Initializer/zeros/shape_as_tensor0conv1_mod/weights/Adam_1/Initializer/zeros/Const*&
_output_shapes
: *
T0*

index_type0*$
_class
loc:@conv1_mod/weights
┬
conv1_mod/weights/Adam_1
VariableV2*
shared_name *$
_class
loc:@conv1_mod/weights*
	container *
shape: *
dtype0*&
_output_shapes
: 
№
conv1_mod/weights/Adam_1/AssignAssignconv1_mod/weights/Adam_1*conv1_mod/weights/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@conv1_mod/weights*
validate_shape(*&
_output_shapes
: 
џ
conv1_mod/weights/Adam_1/readIdentityconv1_mod/weights/Adam_1*
T0*$
_class
loc:@conv1_mod/weights*&
_output_shapes
: 
д
7conv1_mod/biases/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB: *#
_class
loc:@conv1_mod/biases
Ќ
-conv1_mod/biases/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *#
_class
loc:@conv1_mod/biases
з
'conv1_mod/biases/Adam/Initializer/zerosFill7conv1_mod/biases/Adam/Initializer/zeros/shape_as_tensor-conv1_mod/biases/Adam/Initializer/zeros/Const*
_output_shapes
: *
T0*

index_type0*#
_class
loc:@conv1_mod/biases
д
conv1_mod/biases/Adam
VariableV2*
dtype0*
_output_shapes
: *
shared_name *#
_class
loc:@conv1_mod/biases*
	container *
shape: 
┘
conv1_mod/biases/Adam/AssignAssignconv1_mod/biases/Adam'conv1_mod/biases/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@conv1_mod/biases*
validate_shape(*
_output_shapes
: 
Є
conv1_mod/biases/Adam/readIdentityconv1_mod/biases/Adam*
_output_shapes
: *
T0*#
_class
loc:@conv1_mod/biases
е
9conv1_mod/biases/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB: *#
_class
loc:@conv1_mod/biases*
dtype0*
_output_shapes
:
Ў
/conv1_mod/biases/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@conv1_mod/biases*
dtype0*
_output_shapes
: 
щ
)conv1_mod/biases/Adam_1/Initializer/zerosFill9conv1_mod/biases/Adam_1/Initializer/zeros/shape_as_tensor/conv1_mod/biases/Adam_1/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@conv1_mod/biases*
_output_shapes
: 
е
conv1_mod/biases/Adam_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *#
_class
loc:@conv1_mod/biases
▀
conv1_mod/biases/Adam_1/AssignAssignconv1_mod/biases/Adam_1)conv1_mod/biases/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@conv1_mod/biases*
validate_shape(*
_output_shapes
: 
І
conv1_mod/biases/Adam_1/readIdentityconv1_mod/biases/Adam_1*
_output_shapes
: *
T0*#
_class
loc:@conv1_mod/biases
и
8conv2_mod/weights/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"              *$
_class
loc:@conv2_mod/weights*
dtype0*
_output_shapes
:
Ў
.conv2_mod/weights/Adam/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@conv2_mod/weights*
dtype0*
_output_shapes
: 
Ѓ
(conv2_mod/weights/Adam/Initializer/zerosFill8conv2_mod/weights/Adam/Initializer/zeros/shape_as_tensor.conv2_mod/weights/Adam/Initializer/zeros/Const*&
_output_shapes
:  *
T0*

index_type0*$
_class
loc:@conv2_mod/weights
└
conv2_mod/weights/Adam
VariableV2*
dtype0*&
_output_shapes
:  *
shared_name *$
_class
loc:@conv2_mod/weights*
	container *
shape:  
ж
conv2_mod/weights/Adam/AssignAssignconv2_mod/weights/Adam(conv2_mod/weights/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*$
_class
loc:@conv2_mod/weights
ќ
conv2_mod/weights/Adam/readIdentityconv2_mod/weights/Adam*
T0*$
_class
loc:@conv2_mod/weights*&
_output_shapes
:  
╣
:conv2_mod/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"              *$
_class
loc:@conv2_mod/weights*
dtype0*
_output_shapes
:
Џ
0conv2_mod/weights/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@conv2_mod/weights*
dtype0*
_output_shapes
: 
Ѕ
*conv2_mod/weights/Adam_1/Initializer/zerosFill:conv2_mod/weights/Adam_1/Initializer/zeros/shape_as_tensor0conv2_mod/weights/Adam_1/Initializer/zeros/Const*&
_output_shapes
:  *
T0*

index_type0*$
_class
loc:@conv2_mod/weights
┬
conv2_mod/weights/Adam_1
VariableV2*
	container *
shape:  *
dtype0*&
_output_shapes
:  *
shared_name *$
_class
loc:@conv2_mod/weights
№
conv2_mod/weights/Adam_1/AssignAssignconv2_mod/weights/Adam_1*conv2_mod/weights/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@conv2_mod/weights*
validate_shape(*&
_output_shapes
:  
џ
conv2_mod/weights/Adam_1/readIdentityconv2_mod/weights/Adam_1*
T0*$
_class
loc:@conv2_mod/weights*&
_output_shapes
:  
Х
?conv2_mod/BatchNorm/beta/Adam/Initializer/zeros/shape_as_tensorConst*
valueB: *+
_class!
loc:@conv2_mod/BatchNorm/beta*
dtype0*
_output_shapes
:
Д
5conv2_mod/BatchNorm/beta/Adam/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@conv2_mod/BatchNorm/beta*
dtype0*
_output_shapes
: 
Њ
/conv2_mod/BatchNorm/beta/Adam/Initializer/zerosFill?conv2_mod/BatchNorm/beta/Adam/Initializer/zeros/shape_as_tensor5conv2_mod/BatchNorm/beta/Adam/Initializer/zeros/Const*
_output_shapes
: *
T0*

index_type0*+
_class!
loc:@conv2_mod/BatchNorm/beta
Х
conv2_mod/BatchNorm/beta/Adam
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *+
_class!
loc:@conv2_mod/BatchNorm/beta*
	container 
щ
$conv2_mod/BatchNorm/beta/Adam/AssignAssignconv2_mod/BatchNorm/beta/Adam/conv2_mod/BatchNorm/beta/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@conv2_mod/BatchNorm/beta*
validate_shape(*
_output_shapes
: 
Ъ
"conv2_mod/BatchNorm/beta/Adam/readIdentityconv2_mod/BatchNorm/beta/Adam*
T0*+
_class!
loc:@conv2_mod/BatchNorm/beta*
_output_shapes
: 
И
Aconv2_mod/BatchNorm/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB: *+
_class!
loc:@conv2_mod/BatchNorm/beta
Е
7conv2_mod/BatchNorm/beta/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *+
_class!
loc:@conv2_mod/BatchNorm/beta
Ў
1conv2_mod/BatchNorm/beta/Adam_1/Initializer/zerosFillAconv2_mod/BatchNorm/beta/Adam_1/Initializer/zeros/shape_as_tensor7conv2_mod/BatchNorm/beta/Adam_1/Initializer/zeros/Const*
_output_shapes
: *
T0*

index_type0*+
_class!
loc:@conv2_mod/BatchNorm/beta
И
conv2_mod/BatchNorm/beta/Adam_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *+
_class!
loc:@conv2_mod/BatchNorm/beta*
	container *
shape: 
 
&conv2_mod/BatchNorm/beta/Adam_1/AssignAssignconv2_mod/BatchNorm/beta/Adam_11conv2_mod/BatchNorm/beta/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*+
_class!
loc:@conv2_mod/BatchNorm/beta
Б
$conv2_mod/BatchNorm/beta/Adam_1/readIdentityconv2_mod/BatchNorm/beta/Adam_1*
T0*+
_class!
loc:@conv2_mod/BatchNorm/beta*
_output_shapes
: 
и
8conv3_mod/weights/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   *$
_class
loc:@conv3_mod/weights*
dtype0*
_output_shapes
:
Ў
.conv3_mod/weights/Adam/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@conv3_mod/weights*
dtype0*
_output_shapes
: 
Ѓ
(conv3_mod/weights/Adam/Initializer/zerosFill8conv3_mod/weights/Adam/Initializer/zeros/shape_as_tensor.conv3_mod/weights/Adam/Initializer/zeros/Const*&
_output_shapes
: @*
T0*

index_type0*$
_class
loc:@conv3_mod/weights
└
conv3_mod/weights/Adam
VariableV2*
dtype0*&
_output_shapes
: @*
shared_name *$
_class
loc:@conv3_mod/weights*
	container *
shape: @
ж
conv3_mod/weights/Adam/AssignAssignconv3_mod/weights/Adam(conv3_mod/weights/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*$
_class
loc:@conv3_mod/weights
ќ
conv3_mod/weights/Adam/readIdentityconv3_mod/weights/Adam*
T0*$
_class
loc:@conv3_mod/weights*&
_output_shapes
: @
╣
:conv3_mod/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"          @   *$
_class
loc:@conv3_mod/weights
Џ
0conv3_mod/weights/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@conv3_mod/weights*
dtype0*
_output_shapes
: 
Ѕ
*conv3_mod/weights/Adam_1/Initializer/zerosFill:conv3_mod/weights/Adam_1/Initializer/zeros/shape_as_tensor0conv3_mod/weights/Adam_1/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@conv3_mod/weights*&
_output_shapes
: @
┬
conv3_mod/weights/Adam_1
VariableV2*
dtype0*&
_output_shapes
: @*
shared_name *$
_class
loc:@conv3_mod/weights*
	container *
shape: @
№
conv3_mod/weights/Adam_1/AssignAssignconv3_mod/weights/Adam_1*conv3_mod/weights/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@conv3_mod/weights*
validate_shape(*&
_output_shapes
: @
џ
conv3_mod/weights/Adam_1/readIdentityconv3_mod/weights/Adam_1*
T0*$
_class
loc:@conv3_mod/weights*&
_output_shapes
: @
д
7conv3_mod/biases/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:@*#
_class
loc:@conv3_mod/biases*
dtype0*
_output_shapes
:
Ќ
-conv3_mod/biases/Adam/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@conv3_mod/biases*
dtype0*
_output_shapes
: 
з
'conv3_mod/biases/Adam/Initializer/zerosFill7conv3_mod/biases/Adam/Initializer/zeros/shape_as_tensor-conv3_mod/biases/Adam/Initializer/zeros/Const*
_output_shapes
:@*
T0*

index_type0*#
_class
loc:@conv3_mod/biases
д
conv3_mod/biases/Adam
VariableV2*
shared_name *#
_class
loc:@conv3_mod/biases*
	container *
shape:@*
dtype0*
_output_shapes
:@
┘
conv3_mod/biases/Adam/AssignAssignconv3_mod/biases/Adam'conv3_mod/biases/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*#
_class
loc:@conv3_mod/biases
Є
conv3_mod/biases/Adam/readIdentityconv3_mod/biases/Adam*
T0*#
_class
loc:@conv3_mod/biases*
_output_shapes
:@
е
9conv3_mod/biases/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:@*#
_class
loc:@conv3_mod/biases*
dtype0*
_output_shapes
:
Ў
/conv3_mod/biases/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@conv3_mod/biases*
dtype0*
_output_shapes
: 
щ
)conv3_mod/biases/Adam_1/Initializer/zerosFill9conv3_mod/biases/Adam_1/Initializer/zeros/shape_as_tensor/conv3_mod/biases/Adam_1/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@conv3_mod/biases*
_output_shapes
:@
е
conv3_mod/biases/Adam_1
VariableV2*
shared_name *#
_class
loc:@conv3_mod/biases*
	container *
shape:@*
dtype0*
_output_shapes
:@
▀
conv3_mod/biases/Adam_1/AssignAssignconv3_mod/biases/Adam_1)conv3_mod/biases/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@conv3_mod/biases*
validate_shape(*
_output_shapes
:@
І
conv3_mod/biases/Adam_1/readIdentityconv3_mod/biases/Adam_1*
_output_shapes
:@*
T0*#
_class
loc:@conv3_mod/biases
Ф
6fc5_mod/weights/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *"
_class
loc:@fc5_mod/weights*
dtype0*
_output_shapes
:
Ћ
,fc5_mod/weights/Adam/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@fc5_mod/weights*
dtype0*
_output_shapes
: 
ш
&fc5_mod/weights/Adam/Initializer/zerosFill6fc5_mod/weights/Adam/Initializer/zeros/shape_as_tensor,fc5_mod/weights/Adam/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@fc5_mod/weights* 
_output_shapes
:
ђ,ђ
░
fc5_mod/weights/Adam
VariableV2*
dtype0* 
_output_shapes
:
ђ,ђ*
shared_name *"
_class
loc:@fc5_mod/weights*
	container *
shape:
ђ,ђ
█
fc5_mod/weights/Adam/AssignAssignfc5_mod/weights/Adam&fc5_mod/weights/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@fc5_mod/weights*
validate_shape(* 
_output_shapes
:
ђ,ђ
і
fc5_mod/weights/Adam/readIdentityfc5_mod/weights/Adam*
T0*"
_class
loc:@fc5_mod/weights* 
_output_shapes
:
ђ,ђ
Г
8fc5_mod/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *"
_class
loc:@fc5_mod/weights*
dtype0*
_output_shapes
:
Ќ
.fc5_mod/weights/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@fc5_mod/weights*
dtype0*
_output_shapes
: 
ч
(fc5_mod/weights/Adam_1/Initializer/zerosFill8fc5_mod/weights/Adam_1/Initializer/zeros/shape_as_tensor.fc5_mod/weights/Adam_1/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@fc5_mod/weights* 
_output_shapes
:
ђ,ђ
▓
fc5_mod/weights/Adam_1
VariableV2*
	container *
shape:
ђ,ђ*
dtype0* 
_output_shapes
:
ђ,ђ*
shared_name *"
_class
loc:@fc5_mod/weights
р
fc5_mod/weights/Adam_1/AssignAssignfc5_mod/weights/Adam_1(fc5_mod/weights/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@fc5_mod/weights*
validate_shape(* 
_output_shapes
:
ђ,ђ
ј
fc5_mod/weights/Adam_1/readIdentityfc5_mod/weights/Adam_1*
T0*"
_class
loc:@fc5_mod/weights* 
_output_shapes
:
ђ,ђ
Б
5fc5_mod/biases/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:ђ*!
_class
loc:@fc5_mod/biases*
dtype0*
_output_shapes
:
Њ
+fc5_mod/biases/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *!
_class
loc:@fc5_mod/biases
В
%fc5_mod/biases/Adam/Initializer/zerosFill5fc5_mod/biases/Adam/Initializer/zeros/shape_as_tensor+fc5_mod/biases/Adam/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@fc5_mod/biases*
_output_shapes	
:ђ
ц
fc5_mod/biases/Adam
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *!
_class
loc:@fc5_mod/biases*
	container 
м
fc5_mod/biases/Adam/AssignAssignfc5_mod/biases/Adam%fc5_mod/biases/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*!
_class
loc:@fc5_mod/biases
ѓ
fc5_mod/biases/Adam/readIdentityfc5_mod/biases/Adam*
T0*!
_class
loc:@fc5_mod/biases*
_output_shapes	
:ђ
Ц
7fc5_mod/biases/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:ђ*!
_class
loc:@fc5_mod/biases*
dtype0*
_output_shapes
:
Ћ
-fc5_mod/biases/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@fc5_mod/biases*
dtype0*
_output_shapes
: 
Ы
'fc5_mod/biases/Adam_1/Initializer/zerosFill7fc5_mod/biases/Adam_1/Initializer/zeros/shape_as_tensor-fc5_mod/biases/Adam_1/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@fc5_mod/biases*
_output_shapes	
:ђ
д
fc5_mod/biases/Adam_1
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *!
_class
loc:@fc5_mod/biases*
	container 
п
fc5_mod/biases/Adam_1/AssignAssignfc5_mod/biases/Adam_1'fc5_mod/biases/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*!
_class
loc:@fc5_mod/biases
є
fc5_mod/biases/Adam_1/readIdentityfc5_mod/biases/Adam_1*
T0*!
_class
loc:@fc5_mod/biases*
_output_shapes	
:ђ
Ф
6fc6_mod/weights/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"   X   *"
_class
loc:@fc6_mod/weights*
dtype0*
_output_shapes
:
Ћ
,fc6_mod/weights/Adam/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@fc6_mod/weights*
dtype0*
_output_shapes
: 
З
&fc6_mod/weights/Adam/Initializer/zerosFill6fc6_mod/weights/Adam/Initializer/zeros/shape_as_tensor,fc6_mod/weights/Adam/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@fc6_mod/weights*
_output_shapes
:	ђX
«
fc6_mod/weights/Adam
VariableV2*
dtype0*
_output_shapes
:	ђX*
shared_name *"
_class
loc:@fc6_mod/weights*
	container *
shape:	ђX
┌
fc6_mod/weights/Adam/AssignAssignfc6_mod/weights/Adam&fc6_mod/weights/Adam/Initializer/zeros*
T0*"
_class
loc:@fc6_mod/weights*
validate_shape(*
_output_shapes
:	ђX*
use_locking(
Ѕ
fc6_mod/weights/Adam/readIdentityfc6_mod/weights/Adam*
T0*"
_class
loc:@fc6_mod/weights*
_output_shapes
:	ђX
Г
8fc6_mod/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"   X   *"
_class
loc:@fc6_mod/weights
Ќ
.fc6_mod/weights/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *"
_class
loc:@fc6_mod/weights
Щ
(fc6_mod/weights/Adam_1/Initializer/zerosFill8fc6_mod/weights/Adam_1/Initializer/zeros/shape_as_tensor.fc6_mod/weights/Adam_1/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@fc6_mod/weights*
_output_shapes
:	ђX
░
fc6_mod/weights/Adam_1
VariableV2*
dtype0*
_output_shapes
:	ђX*
shared_name *"
_class
loc:@fc6_mod/weights*
	container *
shape:	ђX
Я
fc6_mod/weights/Adam_1/AssignAssignfc6_mod/weights/Adam_1(fc6_mod/weights/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	ђX*
use_locking(*
T0*"
_class
loc:@fc6_mod/weights
Ї
fc6_mod/weights/Adam_1/readIdentityfc6_mod/weights/Adam_1*
T0*"
_class
loc:@fc6_mod/weights*
_output_shapes
:	ђX
б
5fc6_mod/biases/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:X*!
_class
loc:@fc6_mod/biases*
dtype0*
_output_shapes
:
Њ
+fc6_mod/biases/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *!
_class
loc:@fc6_mod/biases
в
%fc6_mod/biases/Adam/Initializer/zerosFill5fc6_mod/biases/Adam/Initializer/zeros/shape_as_tensor+fc6_mod/biases/Adam/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@fc6_mod/biases*
_output_shapes
:X
б
fc6_mod/biases/Adam
VariableV2*!
_class
loc:@fc6_mod/biases*
	container *
shape:X*
dtype0*
_output_shapes
:X*
shared_name 
Л
fc6_mod/biases/Adam/AssignAssignfc6_mod/biases/Adam%fc6_mod/biases/Adam/Initializer/zeros*
T0*!
_class
loc:@fc6_mod/biases*
validate_shape(*
_output_shapes
:X*
use_locking(
Ђ
fc6_mod/biases/Adam/readIdentityfc6_mod/biases/Adam*
T0*!
_class
loc:@fc6_mod/biases*
_output_shapes
:X
ц
7fc6_mod/biases/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:X*!
_class
loc:@fc6_mod/biases*
dtype0*
_output_shapes
:
Ћ
-fc6_mod/biases/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@fc6_mod/biases*
dtype0*
_output_shapes
: 
ы
'fc6_mod/biases/Adam_1/Initializer/zerosFill7fc6_mod/biases/Adam_1/Initializer/zeros/shape_as_tensor-fc6_mod/biases/Adam_1/Initializer/zeros/Const*
_output_shapes
:X*
T0*

index_type0*!
_class
loc:@fc6_mod/biases
ц
fc6_mod/biases/Adam_1
VariableV2*!
_class
loc:@fc6_mod/biases*
	container *
shape:X*
dtype0*
_output_shapes
:X*
shared_name 
О
fc6_mod/biases/Adam_1/AssignAssignfc6_mod/biases/Adam_1'fc6_mod/biases/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@fc6_mod/biases*
validate_shape(*
_output_shapes
:X
Ё
fc6_mod/biases/Adam_1/readIdentityfc6_mod/biases/Adam_1*
T0*!
_class
loc:@fc6_mod/biases*
_output_shapes
:X
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

Adam/beta2Const*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w╠+2
Љ
'Adam/update_conv1_mod/weights/ApplyAdam	ApplyAdamconv1_mod/weightsconv1_mod/weights/Adamconv1_mod/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon:gradients/conv1_mod/Conv2D_grad/tuple/control_dependency_1*
T0*$
_class
loc:@conv1_mod/weights*
use_nesterov( *&
_output_shapes
: *
use_locking( 
Ђ
&Adam/update_conv1_mod/biases/ApplyAdam	ApplyAdamconv1_mod/biasesconv1_mod/biases/Adamconv1_mod/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/conv1_mod/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@conv1_mod/biases*
use_nesterov( *
_output_shapes
: 
а
'Adam/update_conv2_mod/weights/ApplyAdam	ApplyAdamconv2_mod/weightsconv2_mod/weights/Adamconv2_mod/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonIgradients/harmonic_layer/conv2_mod/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
:  *
use_locking( *
T0*$
_class
loc:@conv2_mod/weights
╔
.Adam/update_conv2_mod/BatchNorm/beta/ApplyAdam	ApplyAdamconv2_mod/BatchNorm/betaconv2_mod/BatchNorm/beta/Adamconv2_mod/BatchNorm/beta/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon[gradients/harmonic_layer/conv2_mod/BatchNorm/FusedBatchNorm_grad/tuple/control_dependency_2*
use_locking( *
T0*+
_class!
loc:@conv2_mod/BatchNorm/beta*
use_nesterov( *
_output_shapes
: 
б
'Adam/update_conv3_mod/weights/ApplyAdam	ApplyAdamconv3_mod/weightsconv3_mod/weights/Adamconv3_mod/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonKgradients/harmonic_layer_1/conv3_mod/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@conv3_mod/weights*
use_nesterov( *&
_output_shapes
: @
њ
&Adam/update_conv3_mod/biases/ApplyAdam	ApplyAdamconv3_mod/biasesconv3_mod/biases/Adamconv3_mod/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/harmonic_layer_1/conv3_mod/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@conv3_mod/biases*
use_nesterov( *
_output_shapes
:@
§
%Adam/update_fc5_mod/weights/ApplyAdam	ApplyAdamfc5_mod/weightsfc5_mod/weights/Adamfc5_mod/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/fc5_mod/Tensordot/transpose_1_grad/transpose*
use_nesterov( * 
_output_shapes
:
ђ,ђ*
use_locking( *
T0*"
_class
loc:@fc5_mod/weights
Ш
$Adam/update_fc5_mod/biases/ApplyAdam	ApplyAdamfc5_mod/biasesfc5_mod/biases/Adamfc5_mod/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/fc5_mod/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@fc5_mod/biases*
use_nesterov( *
_output_shapes	
:ђ
Ч
%Adam/update_fc6_mod/weights/ApplyAdam	ApplyAdamfc6_mod/weightsfc6_mod/weights/Adamfc6_mod/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/fc6_mod/Tensordot/transpose_1_grad/transpose*
T0*"
_class
loc:@fc6_mod/weights*
use_nesterov( *
_output_shapes
:	ђX*
use_locking( 
ш
$Adam/update_fc6_mod/biases/ApplyAdam	ApplyAdamfc6_mod/biasesfc6_mod/biases/Adamfc6_mod/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/fc6_mod/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@fc6_mod/biases*
use_nesterov( *
_output_shapes
:X
њ
Adam/mulMulbeta1_power/read
Adam/beta1(^Adam/update_conv1_mod/weights/ApplyAdam'^Adam/update_conv1_mod/biases/ApplyAdam(^Adam/update_conv2_mod/weights/ApplyAdam/^Adam/update_conv2_mod/BatchNorm/beta/ApplyAdam(^Adam/update_conv3_mod/weights/ApplyAdam'^Adam/update_conv3_mod/biases/ApplyAdam&^Adam/update_fc5_mod/weights/ApplyAdam%^Adam/update_fc5_mod/biases/ApplyAdam&^Adam/update_fc6_mod/weights/ApplyAdam%^Adam/update_fc6_mod/biases/ApplyAdam*
T0*#
_class
loc:@conv1_mod/biases*
_output_shapes
: 
Џ
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*#
_class
loc:@conv1_mod/biases*
validate_shape(*
_output_shapes
: 
ћ

Adam/mul_1Mulbeta2_power/read
Adam/beta2(^Adam/update_conv1_mod/weights/ApplyAdam'^Adam/update_conv1_mod/biases/ApplyAdam(^Adam/update_conv2_mod/weights/ApplyAdam/^Adam/update_conv2_mod/BatchNorm/beta/ApplyAdam(^Adam/update_conv3_mod/weights/ApplyAdam'^Adam/update_conv3_mod/biases/ApplyAdam&^Adam/update_fc5_mod/weights/ApplyAdam%^Adam/update_fc5_mod/biases/ApplyAdam&^Adam/update_fc6_mod/weights/ApplyAdam%^Adam/update_fc6_mod/biases/ApplyAdam*
T0*#
_class
loc:@conv1_mod/biases*
_output_shapes
: 
Ъ
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*#
_class
loc:@conv1_mod/biases*
validate_shape(*
_output_shapes
: 
╔
AdamNoOp(^Adam/update_conv1_mod/weights/ApplyAdam'^Adam/update_conv1_mod/biases/ApplyAdam(^Adam/update_conv2_mod/weights/ApplyAdam/^Adam/update_conv2_mod/BatchNorm/beta/ApplyAdam(^Adam/update_conv3_mod/weights/ApplyAdam'^Adam/update_conv3_mod/biases/ApplyAdam&^Adam/update_fc5_mod/weights/ApplyAdam%^Adam/update_fc5_mod/biases/ApplyAdam&^Adam/update_fc6_mod/weights/ApplyAdam%^Adam/update_fc6_mod/biases/ApplyAdam^Adam/Assign^Adam/Assign_1
Ъ
initNoOp^conv1_mod/weights/Assign^conv1_mod/biases/Assign^conv2_mod/weights/Assign ^conv2_mod/BatchNorm/beta/Assign'^conv2_mod/BatchNorm/moving_mean/Assign+^conv2_mod/BatchNorm/moving_variance/Assign^conv3_mod/weights/Assign^conv3_mod/biases/Assign^fc5_mod/weights/Assign^fc5_mod/biases/Assign^fc6_mod/weights/Assign^fc6_mod/biases/Assign^beta1_power/Assign^beta2_power/Assign^conv1_mod/weights/Adam/Assign ^conv1_mod/weights/Adam_1/Assign^conv1_mod/biases/Adam/Assign^conv1_mod/biases/Adam_1/Assign^conv2_mod/weights/Adam/Assign ^conv2_mod/weights/Adam_1/Assign%^conv2_mod/BatchNorm/beta/Adam/Assign'^conv2_mod/BatchNorm/beta/Adam_1/Assign^conv3_mod/weights/Adam/Assign ^conv3_mod/weights/Adam_1/Assign^conv3_mod/biases/Adam/Assign^conv3_mod/biases/Adam_1/Assign^fc5_mod/weights/Adam/Assign^fc5_mod/weights/Adam_1/Assign^fc5_mod/biases/Adam/Assign^fc5_mod/biases/Adam_1/Assign^fc6_mod/weights/Adam/Assign^fc6_mod/weights/Adam_1/Assign^fc6_mod/biases/Adam/Assign^fc6_mod/biases/Adam_1/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
т
save/SaveV2/tensor_namesConst*ў
valueјBІ"Bbeta1_powerBbeta2_powerBconv1_mod/biasesBconv1_mod/biases/AdamBconv1_mod/biases/Adam_1Bconv1_mod/weightsBconv1_mod/weights/AdamBconv1_mod/weights/Adam_1Bconv2_mod/BatchNorm/betaBconv2_mod/BatchNorm/beta/AdamBconv2_mod/BatchNorm/beta/Adam_1Bconv2_mod/BatchNorm/moving_meanB#conv2_mod/BatchNorm/moving_varianceBconv2_mod/weightsBconv2_mod/weights/AdamBconv2_mod/weights/Adam_1Bconv3_mod/biasesBconv3_mod/biases/AdamBconv3_mod/biases/Adam_1Bconv3_mod/weightsBconv3_mod/weights/AdamBconv3_mod/weights/Adam_1Bfc5_mod/biasesBfc5_mod/biases/AdamBfc5_mod/biases/Adam_1Bfc5_mod/weightsBfc5_mod/weights/AdamBfc5_mod/weights/Adam_1Bfc6_mod/biasesBfc6_mod/biases/AdamBfc6_mod/biases/Adam_1Bfc6_mod/weightsBfc6_mod/weights/AdamBfc6_mod/weights/Adam_1*
dtype0*
_output_shapes
:"
Д
save/SaveV2/shape_and_slicesConst*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:"
ј
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerconv1_mod/biasesconv1_mod/biases/Adamconv1_mod/biases/Adam_1conv1_mod/weightsconv1_mod/weights/Adamconv1_mod/weights/Adam_1conv2_mod/BatchNorm/betaconv2_mod/BatchNorm/beta/Adamconv2_mod/BatchNorm/beta/Adam_1conv2_mod/BatchNorm/moving_mean#conv2_mod/BatchNorm/moving_varianceconv2_mod/weightsconv2_mod/weights/Adamconv2_mod/weights/Adam_1conv3_mod/biasesconv3_mod/biases/Adamconv3_mod/biases/Adam_1conv3_mod/weightsconv3_mod/weights/Adamconv3_mod/weights/Adam_1fc5_mod/biasesfc5_mod/biases/Adamfc5_mod/biases/Adam_1fc5_mod/weightsfc5_mod/weights/Adamfc5_mod/weights/Adam_1fc6_mod/biasesfc6_mod/biases/Adamfc6_mod/biases/Adam_1fc6_mod/weightsfc6_mod/weights/Adamfc6_mod/weights/Adam_1*0
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
э
save/RestoreV2/tensor_namesConst"/device:CPU:0*ў
valueјBІ"Bbeta1_powerBbeta2_powerBconv1_mod/biasesBconv1_mod/biases/AdamBconv1_mod/biases/Adam_1Bconv1_mod/weightsBconv1_mod/weights/AdamBconv1_mod/weights/Adam_1Bconv2_mod/BatchNorm/betaBconv2_mod/BatchNorm/beta/AdamBconv2_mod/BatchNorm/beta/Adam_1Bconv2_mod/BatchNorm/moving_meanB#conv2_mod/BatchNorm/moving_varianceBconv2_mod/weightsBconv2_mod/weights/AdamBconv2_mod/weights/Adam_1Bconv3_mod/biasesBconv3_mod/biases/AdamBconv3_mod/biases/Adam_1Bconv3_mod/weightsBconv3_mod/weights/AdamBconv3_mod/weights/Adam_1Bfc5_mod/biasesBfc5_mod/biases/AdamBfc5_mod/biases/Adam_1Bfc5_mod/weightsBfc5_mod/weights/AdamBfc5_mod/weights/Adam_1Bfc6_mod/biasesBfc6_mod/biases/AdamBfc6_mod/biases/Adam_1Bfc6_mod/weightsBfc6_mod/weights/AdamBfc6_mod/weights/Adam_1*
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
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*0
dtypes&
$2"*ъ
_output_shapesІ
ѕ::::::::::::::::::::::::::::::::::
А
save/AssignAssignbeta1_powersave/RestoreV2*
use_locking(*
T0*#
_class
loc:@conv1_mod/biases*
validate_shape(*
_output_shapes
: 
Ц
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*#
_class
loc:@conv1_mod/biases
«
save/Assign_2Assignconv1_mod/biasessave/RestoreV2:2*
use_locking(*
T0*#
_class
loc:@conv1_mod/biases*
validate_shape(*
_output_shapes
: 
│
save/Assign_3Assignconv1_mod/biases/Adamsave/RestoreV2:3*
use_locking(*
T0*#
_class
loc:@conv1_mod/biases*
validate_shape(*
_output_shapes
: 
х
save/Assign_4Assignconv1_mod/biases/Adam_1save/RestoreV2:4*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*#
_class
loc:@conv1_mod/biases
╝
save/Assign_5Assignconv1_mod/weightssave/RestoreV2:5*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@conv1_mod/weights
┴
save/Assign_6Assignconv1_mod/weights/Adamsave/RestoreV2:6*
T0*$
_class
loc:@conv1_mod/weights*
validate_shape(*&
_output_shapes
: *
use_locking(
├
save/Assign_7Assignconv1_mod/weights/Adam_1save/RestoreV2:7*
T0*$
_class
loc:@conv1_mod/weights*
validate_shape(*&
_output_shapes
: *
use_locking(
Й
save/Assign_8Assignconv2_mod/BatchNorm/betasave/RestoreV2:8*
T0*+
_class!
loc:@conv2_mod/BatchNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(
├
save/Assign_9Assignconv2_mod/BatchNorm/beta/Adamsave/RestoreV2:9*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*+
_class!
loc:@conv2_mod/BatchNorm/beta
К
save/Assign_10Assignconv2_mod/BatchNorm/beta/Adam_1save/RestoreV2:10*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*+
_class!
loc:@conv2_mod/BatchNorm/beta
╬
save/Assign_11Assignconv2_mod/BatchNorm/moving_meansave/RestoreV2:11*
use_locking(*
T0*2
_class(
&$loc:@conv2_mod/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
: 
о
save/Assign_12Assign#conv2_mod/BatchNorm/moving_variancesave/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@conv2_mod/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
: 
Й
save/Assign_13Assignconv2_mod/weightssave/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@conv2_mod/weights*
validate_shape(*&
_output_shapes
:  
├
save/Assign_14Assignconv2_mod/weights/Adamsave/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@conv2_mod/weights*
validate_shape(*&
_output_shapes
:  
┼
save/Assign_15Assignconv2_mod/weights/Adam_1save/RestoreV2:15*
T0*$
_class
loc:@conv2_mod/weights*
validate_shape(*&
_output_shapes
:  *
use_locking(
░
save/Assign_16Assignconv3_mod/biasessave/RestoreV2:16*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*#
_class
loc:@conv3_mod/biases
х
save/Assign_17Assignconv3_mod/biases/Adamsave/RestoreV2:17*
use_locking(*
T0*#
_class
loc:@conv3_mod/biases*
validate_shape(*
_output_shapes
:@
и
save/Assign_18Assignconv3_mod/biases/Adam_1save/RestoreV2:18*
T0*#
_class
loc:@conv3_mod/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
Й
save/Assign_19Assignconv3_mod/weightssave/RestoreV2:19*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*$
_class
loc:@conv3_mod/weights
├
save/Assign_20Assignconv3_mod/weights/Adamsave/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@conv3_mod/weights*
validate_shape(*&
_output_shapes
: @
┼
save/Assign_21Assignconv3_mod/weights/Adam_1save/RestoreV2:21*
T0*$
_class
loc:@conv3_mod/weights*
validate_shape(*&
_output_shapes
: @*
use_locking(
Г
save/Assign_22Assignfc5_mod/biasessave/RestoreV2:22*
use_locking(*
T0*!
_class
loc:@fc5_mod/biases*
validate_shape(*
_output_shapes	
:ђ
▓
save/Assign_23Assignfc5_mod/biases/Adamsave/RestoreV2:23*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*!
_class
loc:@fc5_mod/biases
┤
save/Assign_24Assignfc5_mod/biases/Adam_1save/RestoreV2:24*
use_locking(*
T0*!
_class
loc:@fc5_mod/biases*
validate_shape(*
_output_shapes	
:ђ
┤
save/Assign_25Assignfc5_mod/weightssave/RestoreV2:25*
use_locking(*
T0*"
_class
loc:@fc5_mod/weights*
validate_shape(* 
_output_shapes
:
ђ,ђ
╣
save/Assign_26Assignfc5_mod/weights/Adamsave/RestoreV2:26*
T0*"
_class
loc:@fc5_mod/weights*
validate_shape(* 
_output_shapes
:
ђ,ђ*
use_locking(
╗
save/Assign_27Assignfc5_mod/weights/Adam_1save/RestoreV2:27*
validate_shape(* 
_output_shapes
:
ђ,ђ*
use_locking(*
T0*"
_class
loc:@fc5_mod/weights
г
save/Assign_28Assignfc6_mod/biasessave/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@fc6_mod/biases*
validate_shape(*
_output_shapes
:X
▒
save/Assign_29Assignfc6_mod/biases/Adamsave/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@fc6_mod/biases*
validate_shape(*
_output_shapes
:X
│
save/Assign_30Assignfc6_mod/biases/Adam_1save/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@fc6_mod/biases*
validate_shape(*
_output_shapes
:X
│
save/Assign_31Assignfc6_mod/weightssave/RestoreV2:31*
use_locking(*
T0*"
_class
loc:@fc6_mod/weights*
validate_shape(*
_output_shapes
:	ђX
И
save/Assign_32Assignfc6_mod/weights/Adamsave/RestoreV2:32*
validate_shape(*
_output_shapes
:	ђX*
use_locking(*
T0*"
_class
loc:@fc6_mod/weights
║
save/Assign_33Assignfc6_mod/weights/Adam_1save/RestoreV2:33*
T0*"
_class
loc:@fc6_mod/weights*
validate_shape(*
_output_shapes
:	ђX*
use_locking(
╬
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33"П8.