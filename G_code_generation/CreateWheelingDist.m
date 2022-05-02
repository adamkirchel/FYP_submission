function[Strats] = CreateWheelingDist(Params)

% Manufacturing strategy cellular array
Strats = {};

switch Params.DistributionType
    case 'Normal'
        for i = 1:Params.NumberPasses
            amp = normrnd(Params.MeanAmplitude,Params.StandardDev);
            if amp > Params.UpperAmpLim
                Params.Amplitude = Params.LowerAmpLim;
            elseif amp < Params.LowerAmpLim
                Params.Amplitude = Params.LowerAmpLim;
            else
                Params.Amplitude = amp;
            end
            
            Params.NormAmp = Params.Amplitude/Params.SheetSize;
            
            % Determine nodes
            MS = AddNodes(Params);
            MS.setup = Params;
            
            % Calculate effector kinematics
            MS = CalculateKinematics(MS,Params);
            
            Strats{i} = MS;
        end
        
    case 'Uniform'
        
    case 'blah'
        
end