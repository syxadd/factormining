

def build_net(netname: str, option: dict = {} ):
	if netname == "AlphanetV2":
		from .alphanetv2 import AlphaNetv2
		net = AlphaNetv2(**option)
		return net
	elif netname == "AlphanetV1":
		from .network import AlphaNetv1
		net = AlphaNetv1(**option)
		return net
	elif netname == "AlphaNetv2Mod":
		from .alphanetv2lstmall import AlphaNetv2Mod
		return AlphaNetv2Mod(**option)
	elif netname == "AlphaNetv3GRU":
		from .alphanetv3 import AlphaNetv3GRU
		return AlphaNetv3GRUmod(**option)
	elif netname == "AlphaNetv3GRUmod":
		from .alphanetv3mod2 import AlphaNetv3GRUmod
		return AlphaNetv3GRUmod(**option)
	elif netname == "FactorNetLSTM_4encoder":
		from .factornet4enco import FactorNetLSTM_4encoder
		return FactorNetLSTM_4encoder(**option)
	elif netname == "FactorNetLSTM_3encoder":
		from .factornet3enco import FactorNetLSTM_3encoder
		return FactorNetLSTM_3encoder(**option)
	elif netname == "FactorNetLSTM_2encoder":
		from .factornet2enco import FactorNetLSTM_2encoder
		return FactorNetLSTM_2encoder(**option)
	elif netname == "FactorNetLSTM_1encoder":
		from .factornet1enco import FactorNetLSTM_1encoder
		return FactorNetLSTM_1encoder(**option)
	elif netname == "FactorNetLSTM_UNet":
		## 220822
		from .factornetMultiscale220821 import FactorNetLSTM_UNet
		return FactorNetLSTM_UNet(**option)
	elif netname == "FactorNetCombine220904":
		from .factornetCombine220904 import FactorNetCombine220904
		return FactorNetCombine220904(**option)
	elif netname == "FactorNetCombineLN220906":
		from .factornetCombineLN220906 import FactorNetCombineLN220906
		return FactorNetCombineLN220906(**option)
	elif netname == "FactorNetCombine220906small":
		from .factornetCombine220906 import FactorNetCombine220906small
		return FactorNetCombine220906small(**option)
	elif netname == "FactorNetLSTM_3level":
		from .factornetMultilevelFactor220907 import FactorNetLSTM_3level
		return FactorNetLSTM_3level(**option)
	elif netname == "FactorNet_3level220909":
		from .factornetMultilevelFactor220909 import FactorNet_3level220909
		return FactorNet_3level220909(**option)
	elif netname == "FactorNet_3leveWithInputl220909":
		from .factornetMultilevelWithInput220909 import FactorNet_3leveWithInputl220909
		return FactorNet_3leveWithInputl220909(**option)
	elif netname == "FactorNetLSTM_4encoderRemoveLSTM220911":
		from .factornet4enco220911removeLSTM import FactorNetLSTM_4encoderRemoveLSTM220911
		return FactorNetLSTM_4encoderRemoveLSTM220911(**option)
	elif netname == "FactorNetID14_Multiple3Level220911":
		from .factornetID14Multiple220911 import FactorNetID14_Multiple3Level220911
		return FactorNetID14_Multiple3Level220911(**option)
	elif netname == "FactorNetID14_SingleLevel220911":
		from .factornetID14singleLevel import FactorNetID14_SingleLevel220911
		return FactorNetID14_SingleLevel220911(**option)
	elif netname == "FactorNetID15_LargerLevel220911":
		from .factornetID15Multilevel220911 import FactorNetID15_LargerLevel220911
		return FactorNetID15_LargerLevel220911(**option)
	elif netname == "FactorNetID16_LargerLevelLearnable220911":
		from .factornetID16MultilevelAdditionParams220911 import FactorNetID16_LargerLevelLearnable220911
		return FactorNetID16_LargerLevelLearnable220911(**option)
	elif netname == "FactorNetID17_LearnableShareEncoder220913":
		from .factornetID17MultilevelLearnEncoderShareable220911 import FactorNetID17_LearnableShareEncoder220913
		return FactorNetID17_LearnableShareEncoder220913(**option)
	else:
		raise NotImplementedError
