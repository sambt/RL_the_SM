(* ::Package:: *)

(* ::Input::Initialization:: *)
(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)
(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXX THE SYMMETRY TO TENSOR PRODUCTS - PLESTHYSM XXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)
(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Hook length formula for the dimension of the Sn representations. Should be the same as SnClassCharacter[partition,ConstantArray[1,Total[partition]]] *)
SnIrrepDim[partition_]:=Module[{n1,n2,inverseP,result},
n1=partition[[1]];
n2=Length[partition];
inverseP=Count[partition,x_/;x>=#]&/@Range[n1];
result=Table[Max[partition[[j]]+inverseP[[i]]-(j-1)-(i-1)-1,1],{i,n1},{j,n2}];
result=Total[partition]!/(Times@@Flatten[result]);
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* This method decomposes the product of a list of Sn representations into its irreducible parts. It returns a list of integers x_i which are the multiplicity of the irreps IntegerPartitions[n][[i]] in the product of Sn irreps given as input. *)
DecomposeSnProductBase1[partitionsList_]:=Module[{n,characterTable,aux,result},
(* must be the same as for all partitions in partitionsList *)
n=Total[partitionsList[[1]]]; 
result=1/n!Table[Sum[SnClassOrder[i]Product[SnClassCharacter[inputPartition,i],{inputPartition,partitionsList}]SnClassCharacter[j,i],{i,IntegerPartitions[n]}],{j,IntegerPartitions[n]}];
Return[result];
]

(* Added 18/December/2018; Modified 14/January/2020 *)
(* Variation/Generalization of DecomposeSnProductBase1. First, the input are now irrpes of a product of Sn groups: Sn1 x Sn2 x .... Second, the output is a list with elements elI={<Sn1 x Sn2 x ... irrep>, <multiplicity>) *)
DecomposeSnProductBase2[snIrreps_]:=DecomposeSnProductBase2[snIrreps]=Module[{aux,aux2,result},
aux=Transpose[snIrreps];
aux2=Table[Transpose[{IntegerPartitions[Total[el[[1]]]],DecomposeSnProductBase1[el]}],{el,aux}];
result=TuplesWithMultiplicity[DeleteCases[#,{_,0}]&/@aux2];
Return[result];
]

(* Checks if snIrrep is an irrep of a single Sn group (output=True) or rather if it is an irrep of Sn1 x Sn2 x ... (output=False) *)
SingleSnFactorQ[snIrrep_]:=(Depth[snIrrep]==4)

(* Added 14/January/2020. Note that this function takes the name of another one which is now called DecomposeSnProductBase1 *)
DecomposeSnProduct[snIrreps_]:=If[SingleSnFactorQ[snIrreps],DecomposeSnProductBase2[snIrreps],{#[[1,1]],#[[2]]}&/@(DecomposeSnProductBase2[{#}&/@snIrreps])]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* See arXiv:math/0309225v1[math.CO] - this is an auxiliar method to calculate SnClassCharacter *)
(* Note: a partitition is a list of non-increasing positive integers - see http://en.wikipedia.org/wiki/Young_tableau *)
PartitionSequence[partition_]:=Module[{sequence},
sequence=ConstantArray[1,partition[[-1]]];
AppendTo[sequence,0];
Do[
sequence=Join[sequence,ConstantArray[1,partition[[-i]]-partition[[-i+1]]]];
AppendTo[sequence,0];
,{i,2,Length[partition]}];
Return[sequence];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)
(* See arXiv:math/0309225v1[math.CO] - this is an auxiliar method to calculate SnClassCharacter *)
(* This method finds all the rim hooks \[Xi] with length l and returns a list with all the possibilities {partition\\[Xi], leg length of rim hook \[Xi]} which is writen as {partition\\[Xi],ll(\[Xi])}*)
RimHooks[partition_,l_]:=Module[{seqMinusHook,sequence,length,result},
sequence=PartitionSequence[partition];
result={};

Do[
If[sequence[[i]]==1&&sequence[[i+l]]==0,

seqMinusHook=sequence;seqMinusHook[[i]]=0;seqMinusHook[[i+l]]=1;
length=Count[sequence[[i;;i+l]],0]-1;
AppendTo[result,{RebuiltPartitionFromSequence[seqMinusHook],length}];
];

,{i,Length[sequence]-l}];
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)
(* See arXiv:math/0309225v1[math.CO] - this is an auxiliar method to calculate SnClassCharacter *)
(* RebuiltPartitionFromSequence[PartitionSequence[partition]]=partition *)
RebuiltPartitionFromSequence[sequence_]:=Module[{start,end,validSequence,counter1s,result},
counter1s=0;result={};
Do[
If[sequence[[i]]==0,
PrependTo[result,counter1s];
,
counter1s++;
];

,{i,Length[sequence]}];
Return[DeleteCases[result,0]];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)
(* See arXiv:math/0309225v1[math.CO] for the way to compute SnClassCharacter from the Murnaghan-Nakayama rule  *)
(* \[Lambda] is the representation; \[Mu] is the conjugacy class. This method computes the character of conjugacy class \mu in the irreducible representation \[Lambda]  *)
SnClassCharacter[partition\[Lambda]_,partition\[Mu]_]:=SnClassCharacter[partition\[Lambda],partition\[Mu]]=Module[{new\[Lambda]s,new\[Mu],n,result},

If[Length[partition\[Lambda]]==0,Return[1]];

n=Total[partition\[Lambda]];
If[n!=Total[partition\[Mu]],Return["Error in SnClassCharacter function: both partitions must be of the same order."]];

new\[Lambda]s=RimHooks[partition\[Lambda],partition\[Mu][[1]]];
new\[Mu]=partition\[Mu][[2;;-1]];

result=Sum[(-1)^new\[Lambda]s[[i,2]] SnClassCharacter[new\[Lambda]s[[i,1]],new\[Mu]],{i,Length[new\[Lambda]s]}];

Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Size of a given conjugacy class of Sn. The formula is easy but see for example "Enumerative Combinatorics", Richard P.Stanley, http://math.mit.edu/~rstan/ec/ec1.pdf, 1.3.2 Proposition *)
SnClassOrder[partition_]:=SnClassOrder[partition]=Module[{aux,n,result},
n=Total[partition];
aux=Tally[partition];
result=n!/Product[aux[[i,1]]^aux[[i,2]] aux[[i,2]]!,{i,Length[aux]}];
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)
(* The functions VDecomp, AltDom, Adams, Plethysms and ReduceRepPolyProduct below are implementations of the algorithm described in the manual of the program Lie,available at http://www-math.univ-poitiers.fr/~maavl/LiE/ . Except for Plethysms, they are auxiliar functions. *)

VDecomp[cm_,dominantWeight_]:=Module[{result},
result=AltDom[cm,WeylOrbit[cm,dominantWeight]];
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

AltDom[cm_,weights_,weylWord_]:=Module[{cmInv,matD,cmID,aux,prov,result},
prov={#,1}&/@weights;


Do[
(* aux=SimpleProduct[prov[[j]],cm[[i]],cmID]; *)
If[prov[[j,2]]!=0,
Which[prov[[j,1,weylWord[[i]]]]>=0,Null,
prov[[j,1,weylWord[[i]]]]==-1,prov[[j,2]]=0,
prov[[j,1,weylWord[[i]]]]<=-2,prov[[j,2]]=-prov[[j,2]];prov[[j,1]]=prov[[j,1]]-(prov[[j,1,weylWord[[i]]]]+1)cm[[weylWord[[i]]]];

];

];

,{i,Length[weylWord]},{j,Length[prov]}];
prov=DeleteCases[prov,x_/;x[[2]]==0];

Return[prov];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

AltDom[cm_,weights_]:=AltDom[cm,weights,LongestWeylWord[cm]]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

Adams[cm_,n_,rep_]:=Module[{aux,result},
aux=DominantWeights[cm,rep];
aux={#[[1]] n,#[[2]]}&/@aux;

result=Table[{VDecomp[cm,aux[[i,1]]],aux[[i,2]]},{i,Length[aux]}];
result=Table[{result[[i,1,j,1]],result[[i,1,j,2]]result[[i,2]]},{i,Length[result]},{j,Length[result[[i,1]]]}];
result=Flatten[result,1];
Return[result];
]
(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

Options[Plethysms]={UseName->False};
Plethysms[cm_,weightIn_,partition_,OptionsPattern[]]:=Plethysms[cm,weightIn,partition]=Module[{weight,n,kList,aux,aux1,aux2,factor,sum,antiFundamentalQ,result},
weight=SimpleRepInputConversion[cm,weightIn];
n=Plus@@partition;

(* If group = U1 *)
If[cm===U1,
Return[If[Length[partition]==1,{{n weight,1}},{}]];
];

(* If group = SU(n) and weight is the fundamental or anti-fundamental *)
If[Total[weight]==1 &&(weight[[1]]==1||weight[[-1]]==1)&&cm==CartanMatrix["SU",Length[cm]+1],
antiFundamentalQ=!(weight[[1]]==1);

result=ConvertPartitionToDynkinCoef[Length[cm]+1,partition];
If[antiFundamentalQ,result=Reverse[result]];
If[result=!={},result={{result,1}}];

If[OptionValue[UseName],
result=MapThread[List,{RepNameBatchMode[cm,Simplify[result[[All,1]]]],result[[All,2]]}]];
Return[result];
];



kList=IntegerPartitions[n];
sum={};

Do[
factor=1/n!SnClassOrder[kList[[i]]]SnClassCharacter[partition,kList[[i]]];

aux=Adams[cm,#,weight]&/@kList[[i]];

aux=ReduceRepPolyProduct[cm,aux];
aux={#[[1]],factor #[[2]]}&/@aux;

AppendTo[sum,aux];
,{i,Length[kList]}];

sum=GatherWeights[sum];

If[OptionValue[UseName],
sum=MapThread[List,{RepNameBatchMode[cm,Simplify[sum[[All,1]]]],sum[[All,2]]}]];
Return[sum];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)
(* Added 18/December/2018 *)
(* Essentially it uses Plethysms to calculate the all plethysms for all partitions of some n *)

PlethysmsN[groupSimple_,repIn_,n_]:=PlethysmsN[groupSimple,repIn,n]=Module[{rep,partitions,result},
rep=SimpleRepInputConversion[groupSimple,repIn];

partitions=IntegerPartitions[n];
result={Plethysms[groupSimple,rep,#],#}&/@partitions;
result=DeleteCases[result,{{},_}];

result=Table[{result[[i,1,j,1]],result[[i,2]],result[[i,1,j,2]]},{i,Length[result]},{j,Length[result[[i,1]]]}];
result=Flatten[result,1];
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

InvariantPlethysms[cm_,weightIn_,partition_]:=InvariantPlethysms[cm,weightIn,partition]=Module[{weight,n,kList,aux,aux1,aux2,factor,sum},
weight=SimpleRepInputConversion[cm,weightIn];
n=Plus@@partition;
kList=IntegerPartitions[n];
sum=0;

(* If group = U1 *)
If[cm===U1,
Return[If[Length[partition]==1&&E^(I weight)==1,1,0]];
];

Do[
factor=1/n!SnClassOrder[kList[[i]]]SnClassCharacter[partition,kList[[i]]];
aux=Adams[cm,#,weight]&/@kList[[i]];
aux=NumberOfInvariantsInProduct[cm,aux];
sum=sum+factor aux;

,{i,Length[kList]}];

Return[sum];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)
(* This method calculates the decompositions of a product of sums of irreps: (R11+R12+R13+...) x (R21+R22+R23+...) x ... *)
(* polyList = list of lists of representations to be multiplied. The method outputs the decomposition of such a product *)
ReduceRepPolyProduct[cm_,polyList_]:=Module[{n,aux,aux2},

n=Length[polyList];
aux=polyList[[1]];
If[n<=1,Return[aux]];
Do[
aux=Tuples[{aux,polyList[[i+1]]}];

aux2=ReduceRepProduct[cm,#[[1;;2,1]]]&/@aux;

aux=GatherWeights[aux2,#[[1,2]]#[[2,2]]&/@aux];
,{i,n-1}];

Return[aux];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)
(* This method is similar to ReduceRepPolyProduct but it only cares about the number of invariants in product. That is its output *)
NumberOfInvariantsInProduct[cm_,polyList_]:=Module[{n,aux,aux2,invariantRep,list1,list2},
invariantRep=ConstantArray[0,Length[cm]];
n=Length[polyList];

(* If there is no product, just find the number of invariants in polyList[[1]] and return *)
If[n==1,
aux=Cases[polyList[[1]],x_/;x[[1]]==0x[[1]]:>x[[2]]];
Return[Total[aux]];
];

(* If theres is more than one factor list in polyList, break polyList in two parts with abou the same length  *)
(* E.g.: if Length[polyList]=2 then Length[list1]=Length[list2]=1; If Length[polyList]=3 then Length[list1]=1 and Length[list2]=2. *)
(* Conjugate the irreps in the first part (list1) and match then to irreps in second part (list2) - those form invariants *)
list1=ReduceRepPolyProduct[cm,polyList[[1;;Floor[n/2] ]]];
list2=ReduceRepPolyProduct[cm,polyList[[Floor[n/2]+1;;-1]]];
list1={ConjugateIrrep[cm,#[[1]]],#[[2]]}&/@list1;

aux=(Cases[list2,x_/;x[[1]]==#[[1]]:>x[[2]]]&/@list1);
aux=Total[Flatten[list1[[All,2]]aux]];

Return[aux];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Alias of PermutationSymmetryOfTensorProductParts *)
Options[PermutationSymmetry]={DistinguishFields->False,UseName->False};
PermutationSymmetry[groupIn_,listOfRepsIn_,OptionsPattern[]]:=PermutationSymmetryOfTensorProductParts[groupIn,listOfRepsIn,DistinguishFields->OptionValue[DistinguishFields],UseName->OptionValue[UseName]]

(* Updated 18/December/2018 *)
Options[PermutationSymmetryOfTensorProductParts]={DistinguishFields->False,UseName->False};
PermutationSymmetryOfTensorProductParts[groupIn_,listOfRepsIn_,OptionsPattern[]]:=Module[{group,sumCharges,listOfReps,aux1,aux2,aux3,aux4,plesthysmFields,plesthysmFieldsUnderFactorGroups,sizeOfSnSubgroups,orderingX,savedResults,result},
(* Deal with single simple groups *)
If[!IsSimpleGroupQ[groupIn],
group=groupIn;listOfReps=listOfRepsIn;
,
group={groupIn};listOfReps={#}&/@listOfRepsIn;
];

listOfReps=SimpleRepInputConversion[group,#]&/@listOfReps;


(* ************** Distinguish all fields, by temporarily creating a fake hypercharge ************** *)
If[OptionValue[DistinguishFields]===True,
group=Join[group,{U1}];

Do[listOfReps[[i]]=Join[listOfReps[[i]],{i}];,{i,Length[listOfReps]}];
listOfReps[[-1,-1]]=-(1/2) (Length[listOfReps]-1) Length[listOfReps];
];

If[Head[OptionValue[DistinguishFields]]===List,
group=Join[group,{U1}];
aux1=Flatten[Position[OptionValue[DistinguishFields],#]]&/@DeleteDuplicates[OptionValue[DistinguishFields]];

sumCharges=0;
Do[If[i!=Length[aux1],sumCharges+=i;listOfReps[[j]]=Join[listOfReps[[j]],{i}],listOfReps[[j]]=Join[listOfReps[[j]],{-sumCharges/Length[aux1[[i]]]}]],{i,Length[aux1]},{j,aux1[[i]]}];
];
(* ************** Distinguish all fields, by temporarily creating a fake hypercharge ************** *)


aux1=Tally[listOfReps];

(* groups of equal fields: *)
plesthysmFields=Flatten[Position[listOfReps,#]]&/@aux1[[All,1]];

aux2=Transpose[listOfReps];
(* groups of equal fields considering only one factor group at a time: *)
plesthysmFieldsUnderFactorGroups=Table[aux1=Tally[el];Flatten[Position[el,#]]&/@aux1[[All,1]],{el,aux2}];

aux3=Table[Position[plesthysmFieldsUnderFactorGroups[[i]],#][[1,1]]&/@plesthysmFields[[All,1]],{i,Length[plesthysmFieldsUnderFactorGroups]}];
aux4=Table[Flatten[Position[el,#]]&/@DeleteDuplicates[el],{el,aux3}];
orderingX=InvertOrdering/@Flatten/@aux4;
sizeOfSnSubgroups=Map[Length[plesthysmFields[[#]]]&,aux4,{3}];

(*
GENERAL STRATEGY: Compute the symmetries considering only one factor group at a time. This reduces the number of aparent distinct fields. Later, break the irreps of the big Sn symmetry (=permutation group of sizeOfSnSubgroups[[i]]) into
irreps of the actual permutation group (=permutation group of plesthysmFields).
*)

savedResults=Reap[Do[
aux1=Tally[listOfReps[[All,subGIdx]]];
aux1=PlethysmsN[group[[subGIdx]],#[[1]],#[[2]]]&/@aux1;
aux1=Tuples[aux1];
aux1={ReduceRepProduct[group[[subGIdx]],#[[All,1]]],#[[All,2]],Times@@#[[All,3]]}&/@aux1;
aux1=Flatten[Table[{{el2[[1]],el[[2]]},el[[3]]el2[[2]]},{el,aux1},{el2,el[[1]]}],1];

aux2=Table[CalculateSnBranchingRules[el[[1,2,i]],sizeOfSnSubgroups[[subGIdx,i]]],{el,aux1},{i,Length[el[[1,2]]]}];
aux2=Tuples/@aux2;

aux3=Table[{aux1[[i,1,1]],Flatten[aux2[[i,j,All,1]],1][[orderingX[[subGIdx]]]],aux1[[i,2]]Times@@aux2[[i,j,All,2]]},{i,Length[aux1]},{j,Length[aux2[[i]]]}];
aux3=Flatten[aux3,1];
Sow[aux3];
,{subGIdx,Length[group]}]][[2,1]];

aux1={#[[All,1]],DecomposeSnProduct[#[[All,2]]],Times@@#[[All,3]]}&/@Tuples[savedResults];
aux1=Flatten[Table[{#[[1]],el[[1]],el[[2]]#[[3]]},{el,#[[2]]}]&/@aux1,1];

aux1=GatherBy[aux1,#[[1]]&];
aux1=Flatten[Table[TallyWithMultiplicity[{#[[1;;2]],#[[3]]}&/@el],{el,aux1}],1];

(* ************** [REVERT] Distinguish all fields, by temporarily creating a fake hypercharge ************** *)
If[OptionValue[DistinguishFields]===True||Head[OptionValue[DistinguishFields]]===List,
aux1[[All,1,1]]=aux1[[All,1,1,1;;-2]];
];
(* ************** [REVERT] Distinguish all fields, by temporarily creating a fake hypercharge ************** *)

(* Deal with single simple groups *)
If[IsSimpleGroupQ[groupIn],
aux1[[All,1,1]]=aux1[[All,1,1,1]];
];

result=aux1;
SAVE0={group,groupIn,result};
If[OptionValue[UseName],
result[[All,1]]=MapThread[List,{RepNameBatchMode[groupIn,result[[All,1,1]]],Map[DrawYoungDiagramRaster,result[[All,1,2]],{2}]}];
];

Return[{plesthysmFields,result}];
]

(* Updated 18/December/2018 *)
Options[PermutationSymmetryOfInvariants]={DistinguishFields->False,UseName->False};
PermutationSymmetryOfInvariants[groupIn_,listOfRepsIn_,OptionsPattern[]]:=Module[{group,sumCharges,listOfReps,aux1,aux2,aux3,aux4,plesthysmFields,plesthysmFieldsUnderFactorGroups,sizeOfSnSubgroups,orderingX,savedResults},

(* Deal with single simple groups *)
If[!IsSimpleGroupQ[groupIn],
group=groupIn;listOfReps=listOfRepsIn;
,
group={groupIn};listOfReps={#}&/@listOfRepsIn;
];

listOfReps=SimpleRepInputConversion[group,#]&/@listOfReps;


(* ************** Distinguish all fields, by temporarily creating a fake hypercharge ************** *)
If[OptionValue[DistinguishFields]===True,
group=Join[group,{U1}];

Do[listOfReps[[i]]=Join[listOfReps[[i]],{i}];,{i,Length[listOfReps]}];
listOfReps[[-1,-1]]=-(1/2) (Length[listOfReps]-1) Length[listOfReps];
];

If[Head[OptionValue[DistinguishFields]]===List,
group=Join[group,{U1}];
aux1=Flatten[Position[OptionValue[DistinguishFields],#]]&/@DeleteDuplicates[OptionValue[DistinguishFields]];

sumCharges=0;
Do[If[i!=Length[aux1],sumCharges+=i;listOfReps[[j]]=Join[listOfReps[[j]],{i}],listOfReps[[j]]=Join[listOfReps[[j]],{-sumCharges/Length[aux1[[i]]]}]],{i,Length[aux1]},{j,aux1[[i]]}];
];
(* ************** Distinguish all fields, by temporarily creating a fake hypercharge ************** *)


aux1=Tally[listOfReps];

(* groups of equal fields: *)
plesthysmFields=Flatten[Position[listOfReps,#]]&/@aux1[[All,1]];

aux2=Transpose[listOfReps];
(* groups of equal fields considering only one factor group at a time: *)
plesthysmFieldsUnderFactorGroups=Table[aux1=Tally[el];Flatten[Position[el,#]]&/@aux1[[All,1]],{el,aux2}];

aux3=Table[Position[plesthysmFieldsUnderFactorGroups[[i]],#][[1,1]]&/@plesthysmFields[[All,1]],{i,Length[plesthysmFieldsUnderFactorGroups]}];
aux4=Table[Flatten[Position[el,#]]&/@DeleteDuplicates[el],{el,aux3}];
orderingX=InvertOrdering/@Flatten/@aux4;
sizeOfSnSubgroups=Map[Length[plesthysmFields[[#]]]&,aux4,{3}];

(*
GENERAL STRATEGY: Compute the symmetries considering only one factor group at a time. This reduces the number of aparent distinct fields. Later, break the irreps of the big Sn symmetry (=permutation group of sizeOfSnSubgroups[[i]]) into
irreps of the actual permutation group (=permutation group of plesthysmFields).
*)

savedResults=Reap[Do[
aux1=Tally[listOfReps[[All,subGIdx]]];
aux1=PlethysmsN[group[[subGIdx]],#[[1]],#[[2]]]&/@aux1;
aux1=Tuples[aux1];
(* Cases[AAAAAAAAA,x_/;x[[1]]===0x[[1]]] is what picks only the gauge invariants, dramatically simplifying the follow up calculations *)
aux1={Cases[ReduceRepProduct[group[[subGIdx]],#[[All,1]]],x_/;x[[1]]===0x[[1]]],#[[All,2]],Times@@#[[All,3]]}&/@aux1;
aux1=Flatten[Table[{{el2[[1]],el[[2]]},el[[3]]el2[[2]]},{el,aux1},{el2,el[[1]]}],1];

aux2=Table[CalculateSnBranchingRules[el[[1,2,i]],sizeOfSnSubgroups[[subGIdx,i]]],{el,aux1},{i,Length[el[[1,2]]]}];
aux2=Tuples/@aux2;

aux3=Table[{aux1[[i,1,1]],Flatten[aux2[[i,j,All,1]],1][[orderingX[[subGIdx]]]],aux1[[i,2]]Times@@aux2[[i,j,All,2]]},{i,Length[aux1]},{j,Length[aux2[[i]]]}];
aux3=Flatten[aux3,1];
Sow[aux3];
,{subGIdx,Length[group]}]][[2,1]];

aux1={#[[All,1]],DecomposeSnProduct[#[[All,2]]],Times@@#[[All,3]]}&/@Tuples[savedResults];
aux1=Flatten[Table[{#[[1]],el[[1]],el[[2]]#[[3]]},{el,#[[2]]}]&/@aux1,1];

aux1=GatherBy[aux1,#[[1]]&];
aux1=Flatten[Table[TallyWithMultiplicity[{#[[1;;2]],#[[3]]}&/@el],{el,aux1}],1];

(* ************** [REVERT] Distinguish all fields, by temporarily creating a fake hypercharge ************** *)
If[OptionValue[DistinguishFields]===True||Head[OptionValue[DistinguishFields]]===List,
aux1[[All,1,1]]=aux1[[All,1,1,1;;-2]];
];
(* ************** [REVERT] Distinguish all fields, by temporarily creating a fake hypercharge ************** *)

(* Deal with single simple groups *)
If[IsSimpleGroupQ[groupIn],
aux1[[All,1,1]]=aux1[[All,1,1,1]];
];

aux2=Transpose[{aux1[[All,1,2]],aux1[[All,2]]}];

If[OptionValue[UseName],
aux2={DrawYoungDiagramRaster/@#[[1]],#[[2]]}&/@aux2;
];
Return[{plesthysmFields,aux2}];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Hook-content formula: counts the number of semi-standard Young tableaux of shape given by partition and with the cells filled with the numbers 1,...,n *)
HookContentFormula[partition_,nMax_]:=Module[{n1,n2,inverseP,hookLengths,content,result,aux},
n1=partition[[1]];
n2=Length[partition];
inverseP=Count[partition,x_/;x>=#]&/@Range[n1];

aux=Table[If[partition[[j]]+inverseP[[i]]-(j-1)-(i-1)-1>0,(nMax+i-j)/(partition[[j]]+inverseP[[i]]-(j-1)-(i-1)-1),1],{i,n1},{j,n2}];
result=Times@@Flatten[aux];
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

TransposeTableaux[tableaux_]:=DeleteCases[Transpose[PadRight[#,Length[tableaux[[1]]],Null]&/@tableaux],Null,-1]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Checks if a given tableaux is standard *)
CheckStandardTableaux[tableaux_]:=Module[{transpose},
transpose=TransposeTableaux[tableaux];
Return[And@@Join[OrderedQ/@tableaux,OrderedQ/@transpose]];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

TransposePartition[partition_]:=TransposePartition[partition]=Module[{n1,n2,inverseP},
If[partition==={},Return[{}]];
n1=partition[[1]];
n2=Length[partition];
inverseP=Count[partition,x_/;x>=#]&/@Range[n1];
Return[inverseP];
]

(* Returns a Young tableaux with the entries filled with the Hook length of each square *)
HookLengths[partition_]:=Module[{partitionT,result},
partitionT=TransposePartition[partition];
result=Table[partition[[i]]+partitionT[[j]]-(i-1)-(j-1)-1,{i,Length[partition]},{j,partition[[i]]}];
Return[result];
]

(* Returns a Young tableaux with the entries filled with the maximum entry in each square, if the tableaux is to be standard *)
MaxIndex[partition_]:=Module[{partitionT,result},
partitionT=TransposePartition[partition];
result=Table[
Total[partition]-Total[partition[[1;;i-1]]]-Total[partitionT[[1;;j-1]]]+(i-1)(j-1)
,{i,Length[partition]},{j,partition[[i]]}];
result=Total[partition]+1-result;
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Draws a Young Tableaux *)
DrawTableaux[tableaux_]:=Module[{\[Lambda],\[Lambda]T,gridLines,result},
\[Lambda]=Length/@tableaux;
\[Lambda]T=TransposePartition[\[Lambda]];
gridLines=Join[Table[{{i,i},{1,\[Lambda][[i]]}}->True,{i,Length[\[Lambda]]}],Table[{{1,\[Lambda]T[[i]]},{i,i}}->True,{i,Length[\[Lambda]T]}]];

result=Grid[tableaux,Frame->{None,None,gridLines}];
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Generates all Standard Tableaux with a given shape *)
Options[GenerateStandardTableaux]={Draw->False};
GenerateStandardTableaux[\[Lambda]_,OptionsPattern[]]:=Module[{n,\[Lambda]T,canonicalTableaux,canonicalTableauxT,affectedByList,minIndTable,maxIndTable,result},
If[\[Lambda]==={},Return[{}]];

n=Total[\[Lambda]];
\[Lambda]T=TransposePartition[\[Lambda]];
canonicalTableaux=MapThread[Range,{Accumulate[\[Lambda]]-\[Lambda]+1,Accumulate[\[Lambda]]}];
canonicalTableauxT=TransposeTableaux[canonicalTableaux];

affectedByList=ConstantArray[{},n];

Do[AppendTo[affectedByList[[canonicalTableaux[[i,j-1]]]],canonicalTableaux[[i,j]]],{i,1,Length[\[Lambda]]},{j,2,\[Lambda][[i]]}];
Do[AppendTo[affectedByList[[canonicalTableauxT[[i,j-1]]]],canonicalTableauxT[[i,j]]],{i,1,Length[\[Lambda]T]},{j,2,\[Lambda]T[[i]]}];

maxIndTable=Flatten[MaxIndex[\[Lambda]]];
minIndTable=ConstantArray[1,n];

result=GenerateStandardTableauxAux[ConstantArray[0,n],1,affectedByList,minIndTable,maxIndTable];
result=Transpose[MapThread[result[[All,#1;;#2]]&,{Accumulate[\[Lambda]]-\[Lambda]+1,Accumulate[\[Lambda]]}]];

If[OptionValue[Draw],result=DrawTableaux/@result];
Return[result];
]

(* Auxiliar function to GenerateStandardTableaux *)
GenerateStandardTableauxAux[incompleteList_,idxToFill_,affectedByList_,minIndTable_,maxIndTable_]:=Module[{i,possibleFillings,aux,possibities,minIndTables,result},
If[Abs[maxIndTable-minIndTable]=!=maxIndTable-minIndTable,Return[{}]];

possibleFillings=Complement[Range[minIndTable[[idxToFill]],maxIndTable[[idxToFill]]],incompleteList[[1;;idxToFill-1]]];
possibities=Table[aux=incompleteList;aux[[idxToFill]]=i;aux,{i,possibleFillings}];
minIndTables=Table[aux=minIndTable;aux[[affectedByList[[idxToFill]]]]=Table[Max[aux[[el]],i+1],{el,affectedByList[[idxToFill]]}];aux,{i,possibleFillings}];

If[idxToFill==Length[incompleteList],
Return[possibities];
,
result=Table[GenerateStandardTableauxAux[possibities[[i]],idxToFill+1,affectedByList,minIndTables[[i]],maxIndTable],{i,Length[possibleFillings]}];
Return[Flatten[result,1]];
];
]


(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* See for example "Young tableaux and the representations of the symmetric group", by Yufei Zhao, for a review of the main idea *)
(* Function returns the matrices {P12,P12...n} which generate Sn *)
(*
Subscript[t, i]   \[LongLeftArrow]  Standard Young tableaux of a given shape;
{t}  \[LongLeftArrow]  Tabloid corresponding to a tableaux t
Subscript[e, t]  \[LongLeftArrow]  Polytabloid associated to a tableaux t. By definition, Subscript[e, t] is given by ([Sum over \[Pi] in column group of t] sgn(\[Pi])\[Pi]({t}))

The Subscript[e, Subscript[t, i]] are a basis for the vector space where the irreducible representation acts: Subscript[\[Pi], \[Alpha]]Subscript[e, Subscript[t, i]]=Subscript[(M^\[Alpha]), ji]Subscript[e, Subscript[t, i]] and we want to get the M^\[Alpha] matrices, for \[Alpha]=(12),(12...n). Note that Subscript[\[Pi], \[Alpha]]Subscript[e, Subscript[t, i]]= Subscript[e, Subscript[\[Pi], \[Alpha]](Subscript[t, i])]. The problem is that the Subscript[e, t] might involve a big sum with many {t'} terms. This can be solved by tracking only a few independent ones. For example, {Subscript[\[Pi], \[Alpha]](Subscript[t, i])} will work. So, if X^\[Alpha] and Y^\[Alpha] are such that Subscript[e, Subscript[\[Pi], \[Alpha]](Subscript[t, i])]\[Congruent]Subscript[(X^\[Alpha]), ij] {Subscript[\[Pi], \[Alpha]](Subscript[t, j])}+(other {t'} terms) and Subscript[e, Subscript[t, i]]\[Congruent]Subscript[(Y^\[Alpha]), ij] {Subscript[\[Pi], \[Alpha]](Subscript[t, j])}+(other {t'} terms), then M^\[Alpha]=([X^\[Alpha](Y^\[Alpha])^-1]^T).
*)
Options[SnIrrepGenerators]={OrthogonalizeGenerators->True};
SnIrrepGenerators[\[Lambda]_,OptionsPattern[]]:=Module[{startingTableaux,targetTabloid,Id,n,sts,basicPermutations,tabloids,aux,result,X,Y,ns,BcB,B},
n=Total[\[Lambda]];
sts=GenerateStandardTableaux[\[Lambda]];
basicPermutations={{1->2,2->1},MapThread[Rule,{Range[n],RotateLeft[Range[n]]}]};

tabloids=Table[DeleteDuplicates[Map[Sort,(sts/.\[Alpha]),{2}]],{\[Alpha],basicPermutations}];

X=ConstantArray[Null,{2,Length[sts],Length[tabloids[[1]]]}];
Y=ConstantArray[Null,{2,Length[sts],Length[tabloids[[1]]]}];

Do[
startingTableaux=sts[[i]];
targetTabloid=tabloids[[\[Alpha]I,j]];

Y[[\[Alpha]I,i,j]]=Table[Position[targetTabloid,#][[1,1]]&/@el,{el,TransposeTableaux[startingTableaux]}];
Y[[\[Alpha]I,i,j]]=Times@@Table[If[Sort[el]=!=Range[Length[el]],0,Signature[el]],{el,Y[[\[Alpha]I,i,j]]}];

,{\[Alpha]I,2},{i,Length[sts]},{j,Length[tabloids[[\[Alpha]I]]]}];

Do[
startingTableaux=sts[[i]]/.basicPermutations[[\[Alpha]I]];
targetTabloid=tabloids[[\[Alpha]I,j]];

X[[\[Alpha]I,i,j]]=Table[Position[targetTabloid,#][[1,1]]&/@el,{el,TransposeTableaux[startingTableaux]}];
X[[\[Alpha]I,i,j]]=Times@@Table[If[Sort[el]=!=Range[Length[el]],0,Signature[el]],{el,X[[\[Alpha]I,i,j]]}];

,{\[Alpha]I,2},{i,Length[sts]},{j,Length[tabloids[[\[Alpha]I]]]}];

result=Table[Transpose[X[[i]] . Inverse[Y[[i]]]],{i,2}];


If[!OptionValue[OrthogonalizeGenerators],Return[result]];

(* Orthogonalize the output generators Pi. This is done in the following way
- Oi=B.Pi.Inverse[B] where the Oi are orthogonal, and B is the change of basis matrix;
  - Since the Pi are real, Pi^T.(B^T.B).Pi=B^T.B;
  - If both Pi are taken into consideration, this fixes completely B^T.B as the Pi are generators of the group, in an irreducible representation;
  - With KroneckerProduct and NullSpace, B^T.B can be found, and B can be obtained with CholeskyTypeDecomposition.   *)
result=SparseArray[#,Dimensions[#]]&/@result;
Id=SparseArray[Array[{#,#}->1&,Length[result[[1]]]^2]];

aux=Transpose[KroneckerProduct[Conjugate[#],#]]&/@result;
ns=NullSpace[Join[aux[[1]]-Id,aux[[2]]-Id]][[1]];
BcB=InverseFlatten[ns,{Length[result[[1]]],Length[result[[1]]]}];
B=Transpose[CholeskyTypeDecomposition[BcB]];

result=Simplify[B . # . Inverse[B]]&/@result;
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Auxilar function to LittlewoodRichardsonCoefficients. Converts SU(n) weights into partitions *) 

(* Reverse and Sort are just to make sure that things will work ok even if DeleteCases reorders elements *)
ConvertToPartitionNotation[repr_]:=Reverse[Sort[DeleteCases[Reverse[Accumulate[Reverse[repr]]],0]]]

(* Auxilar function to LittlewoodRichardsonCoefficients. Converts partitions into SU(n) weights *) 
ConvertPartitionToDynkinCoef[n_,partitionIn_]:=Module[{partitionT,partition,aux1,aux2,result},
partitionT=TransposePartition[partitionIn];
If[MemberQ[partitionT,x_/;x>n],Return[{}]];

partition=TransposePartition[DeleteCases[partitionT,n]];
aux1=PadRight[partition,n-1];
aux1=aux1[[1;;n-1]];
aux2=RotateLeft[aux1];
aux2[[-1]]=0;
result=aux1-aux2;
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Calculates the LittlewoodRichardsonCoefficients coefficients using  ReduceRepProduct applied to SU(N) for a sufficiently large N *)
LittlewoodRichardsonCoefficients[\[Lambda]1_,\[Lambda]2_]:=Module[{n,rep1,rep2,result},
n=Length[\[Lambda]1]+Length[\[Lambda]2]+1;
rep1=ConvertPartitionToDynkinCoef[n,\[Lambda]1];
rep2=ConvertPartitionToDynkinCoef[n,\[Lambda]2];
result=ReduceRepProduct[CartanMatrix["SU",n],{rep1,rep2}];
result={ConvertToPartitionNotation[#[[1]]],#[[2]]}&/@result;
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Generatilizes LittlewoodRichardsonCoefficients[\[Lambda]1,\[Lambda]2] to a product of any number of \[Lambda]s *)
LittlewoodRichardsonCoefficients[\[Lambda]s_]:=Module[{n,reps,result},
n=Total[Length/@\[Lambda]s]+1;
reps=ConvertPartitionToDynkinCoef[n,#]&/@\[Lambda]s;
result=ReduceRepProduct[CartanMatrix["SU",n],reps];
result={ConvertToPartitionNotation[#[[1]]],#[[2]]}&/@result;
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Alias of CalculateSnBranchingRules *)
SnBranchingRules[nsOfSubgroups_]:=CalculateSnBranchingRules[nsOfSubgroups]

(* Input is the list of n's of the Sn subgroups: H=Sn1 x Sn2 x Sn3 x... The output is a {<list of G=SN irreps>,<decomposition of G irreps into H irreps>}. All G irreps are decomposed! Note that CalculateSnBranchingRules[\[Lambda],ns] is more targetted, as a specific \[Lambda] can be chosen *)
CalculateSnBranchingRules[nsOfSubgroups_]:=Module[{n,allSubgroupIrreps,allGroupIrreps,aux,aux2,aux3,allGroupIrreps2,subgroupIrreps},
n=Total[nsOfSubgroups];
allSubgroupIrreps=Tuples[IntegerPartitions/@nsOfSubgroups];
allGroupIrreps=IntegerPartitions[n];
aux=LittlewoodRichardsonCoefficients/@allSubgroupIrreps;
aux2=Flatten[Table[{#[[1]],#[[2]],allSubgroupIrreps[[i]]}&/@aux[[i]],{i,Length[allSubgroupIrreps]}],1];
aux3=GatherBy[aux2,#[[1]]&];

allGroupIrreps2=aux3[[All,1,1]];
subgroupIrreps=aux3[[All,All,{3,2}]];
Return[{allGroupIrreps2,subgroupIrreps}];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)
(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)
(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Calculates the number of skew-tableaux \[Lambda]/\[Mu] with a given weight \[Nu] according to the algorithm 1 in https://arxiv.org/pdf/0812.0435.pdf *)
LRSkewTableaux[\[Lambda]_,\[Mu]_]:=Module[{mat,result},
If[Length[\[Mu]]>Length[\[Lambda]],Return[{}]];
If[Abs[\[Lambda]-PadRight[\[Mu],Length[\[Lambda]]]]=!=(\[Lambda]-PadRight[\[Mu],Length[\[Lambda]]]),Return[{}]];
If[Length[\[Lambda]]==1,Return[{\[Lambda]-\[Mu]}]];


mat=ConstantArray[0,{Length[\[Lambda]],\[Lambda][[1]]}];
Do[(mat[[i,1;;\[Lambda][[i]]]]=1+mat[[i,1;;\[Lambda][[i]]]]),{i,Length[\[Lambda]]}];
Do[(mat[[i,1;;\[Mu][[i]]]]=0mat[[i,1;;\[Mu][[i]]]]),{i,Length[\[Mu]]}];

result=Reap[LRSkewTableauxAux[mat]][[2,1]];
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Auxilar function to LRSkewTableaux *)
LRSkewTableauxAux[mat_]:=Module[{n,li,ri,line,matA,matB,aux},
n=Length[mat[[1]]];

li=Flatten[Flatten[Position[#,1,{1},1]]&/@mat /.{}->{n+1}];
ri=(n+1)-Flatten[Flatten[Position[Reverse[#],1,{1},1]]&/@mat /.{}->{n+1}];

line=0;
Do[If[li[[i]]<li[[i-1]],line=i;Break[]],{i,Length[li],2,-1}];

If[line>0,

(* STEP A *)
If[ri[[line-1]]==0||ri[[line]]>=li[[line-1]]-1,
matA=mat;
Do[
aux=matA[[line,j]];
matA[[line,j]]=matA[[line-1,j]];
matA[[line-1,j]]=aux;
,{j,li[[line]],li[[line-1]]-1}];

LRSkewTableauxAux[matA];
];

(* STEP B *)
If[ri[[line]]<=ri[[line-1]]-1,
matB=mat;
Do[
If[li[[line]]!=li[[j]],Break[]];
aux=matB[[j,li[[j]]]];
matB[[j,li[[j]]]]=matB[[j,ri[[j]]+1]];
matB[[j,ri[[j]]+1]]=aux;
,{j,line,Length[mat]}];

LRSkewTableauxAux[matB];

];
,

Sow[DeleteCases[Count[#,1]&/@mat,0]];
];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Auxilar function to CalculateSnBranchingRules[\[Lambda],ns]. It is probabably a good idea to save the values of this function, to improve speed *)
CalculateSnBranchingRulesAux[\[Lambda]_,nOfSubgroups_]:=Module[{\[Lambda]s,aux,result,partitionsSubgroup},
partitionsSubgroup=IntegerPartitions[nOfSubgroups];
aux={#,Tally[LRSkewTableaux[\[Lambda],#]]}&/@partitionsSubgroup;
aux=DeleteCases[aux,{_,{}}];
result=Table[{{aux[[i,1]],aux[[i,2,j,1]]},aux[[i,2,j,2]]},{i,Length[aux]},{j,Length[aux[[i,2]]]}];
result=Flatten[result,1];
Return[result];
]

(* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *)

(* Alias of CalculateSnBranchingRules *)
SnBranchingRules[\[Lambda]in_,nsOfSubgroups_]:=CalculateSnBranchingRules[\[Lambda]in,nsOfSubgroups]

(* Input: Irrep \[Lambda] of some SN to decompose, and nsOfSubgroups={n1,n2,n3,...} indicating the permutation subgroup. Output: list of subgroup irreps in \[Lambda]. Note that this function CalculateSnBranchingRules[\[Lambda],ns] is more targeted than CalculateSnBranchingRules[ns] which must calculate simultaneously the branching rules of all irreps of SN, because it is using ReduceRepProduct *)
CalculateSnBranchingRules[\[Lambda]in_,nsOfSubgroups_]:=CalculateSnBranchingRules[\[Lambda]in,nsOfSubgroups]=Module[{\[Lambda]s,\[Lambda],result,ns,aux,partitionsSubgroup},
If[Length[nsOfSubgroups]==1,Return[{{{\[Lambda]in},1}}]];
\[Lambda]s=If[Depth[\[Lambda]in]==2,{{{\[Lambda]in},1}},\[Lambda]in];
(* ns=Sort[nsOfSubgroups]; *)
ns=nsOfSubgroups;

result=Reap[Do[
\[Lambda]=\[Lambda]s[[i]];
partitionsSubgroup=IntegerPartitions[Total[ns[[2;;-1]]]];
aux=CalculateSnBranchingRulesAux[\[Lambda][[1,-1]],Total[ns[[2;;-1]]]];
aux={Join[\[Lambda][[1,1;;-2]],Reverse[#[[1]]]],#[[2]]\[Lambda][[2]]}&/@aux;
Sow[aux];
,{i,Length[\[Lambda]s]}]][[2,1]];
result=TallyWithMultiplicity[Flatten[result,1]];
If[Length[nsOfSubgroups]>2,
result=CalculateSnBranchingRules[result,ns[[2;;-1]]];
];
Return[result];
]


(* ::Input::Initialization:: *)
(* Alias of DrawYoungDiagram *)
Options[YoungDiagram]={ScaleFactor->20};
YoungDiagram[partition_,OptionsPattern[]]:=DrawYoungDiagram[partition,ScaleFactor->OptionValue[ScaleFactor]]

(* Draws the Young diagram with the associated partition \[Lambda] *)
Options[DrawYoungDiagram]={ScaleFactor->20};
DrawYoungDiagram[\[Lambda]_,OptionsPattern[]]:=Module[{t\[Lambda],horizontalLines,verticalLines,result},
If[\[Lambda]==={},Return[Graphics[{},ImageSize->0]]];
t\[Lambda]=TransposePartition[\[Lambda]];

horizontalLines=Table[Line[{{0,-i},{\[Lambda][[i]],-i}}],{i,Length[\[Lambda]]}];
PrependTo[horizontalLines,Line[{{0,0},{\[Lambda][[1]],0}}]];
verticalLines=Table[Line[{{i,0},{i,-t\[Lambda][[i]]}}],{i,Length[t\[Lambda]]}];
PrependTo[verticalLines,Line[{{0,0},{0,-t\[Lambda][[1]]}}]];
result=Graphics[Join[horizontalLines,verticalLines],ImageSize->(OptionValue[ScaleFactor]Length[t\[Lambda]]),ImagePadding->None,ImageMargins->0,PlotRange->{{0,Length[t\[Lambda]]+0.2},{-(Length[\[Lambda]]+0.2),0}}];

Return[result];
]

(* Leave a 1-pixel border around the picture, because Mathematica sometimes crops the images *)
DrawYoungDiagramRaster[\[Lambda]_,scaleFactor_:9]:=DrawYoungDiagramRaster[\[Lambda],scaleFactor]=Module[{t\[Lambda],horizontalLines,verticalLines,result,nH,nV,dataArray},
If[\[Lambda]==={},Return[Null]];
t\[Lambda]=TransposePartition[\[Lambda]];

nH=scaleFactor Length[t\[Lambda]]+1;
nV=scaleFactor Length[\[Lambda]]+1;

horizontalLines=Table[{{0,-i},{\[Lambda][[i]],-i}},{i,Length[\[Lambda]]}];
PrependTo[horizontalLines,{{0,0},{\[Lambda][[1]],0}}];
verticalLines=Table[{{i,0},{i,-t\[Lambda][[i]]}},{i,Length[t\[Lambda]]}];
PrependTo[verticalLines,{{0,0},{0,-t\[Lambda][[1]]}}];

horizontalLines=scaleFactor horizontalLines;
verticalLines=scaleFactor verticalLines;

dataArray=ConstantArray[1,2+{nV,nH}]; (* leave 1 pixel border *)
Do[
dataArray[[2-hl[[1,2]],2+hl[[1,1]];;2+hl[[2,1]]]]=0dataArray[[2-hl[[1,2]],2+hl[[1,1]];;2+hl[[2,1]]]];
,{hl,horizontalLines}];
Do[
dataArray[[2-vl[[1,2]];;2-vl[[2,2]],2+vl[[1,1]]]]=0dataArray[[2-vl[[1,2]];;2-vl[[2,2]],2+vl[[1,1]]]];
,{vl,verticalLines}];

(* leave 1 pixel border *)
result=SetAlphaChannel[Image[dataArray,ImageSize->2+{nH,nV}],Image[1-dataArray,ImageSize->2+{nH,nV}]];
Return[result];
]
